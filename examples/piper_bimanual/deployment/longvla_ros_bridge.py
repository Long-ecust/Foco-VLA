#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import logging
import threading
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from openpi_client import websocket_client_policy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState


PER_ARM_JOINTS = 7
NUM_JOINTS_TOTAL = 14
FORCE_WINDOW = 16

ROS_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

DEFAULT_VELOCITY_LIMIT = 5.2


class StreamActionBuffer:
    def __init__(self, min_smooth_steps: int = 8):
        self.cur_chunk = collections.deque()
        self.k = 0
        self.last_action: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.min_smooth_steps = int(min_smooth_steps)

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int = 0):
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return

            actions_chunk = np.asarray(actions_chunk, dtype=np.float32)
            if actions_chunk.ndim == 1:
                actions_chunk = actions_chunk[None, :]

            drop_n = min(self.k, int(max_k))
            if drop_n >= len(actions_chunk):
                return

            new_list = [a.copy() for a in actions_chunk[drop_n:]]
            old_list = list(self.cur_chunk)

            if len(old_list) == 0:
                if self.last_action is not None:
                    old_list = [self.last_action.copy() for _ in range(self.min_smooth_steps)]
                else:
                    self.cur_chunk = collections.deque(new_list)
                    self.k = 0
                    return

            if len(old_list) < self.min_smooth_steps:
                old_list += [old_list[-1].copy() for _ in range(self.min_smooth_steps - len(old_list))]

            overlap_len = min(len(old_list), len(new_list))

            if overlap_len <= 0:
                self.cur_chunk = collections.deque(new_list)
                self.k = 0
                return

            w_old = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32)
            w_new = 1.0 - w_old

            smoothed = [
                w_old[i] * np.asarray(old_list[i], dtype=np.float32)
                + w_new[i] * np.asarray(new_list[i], dtype=np.float32)
                for i in range(overlap_len)
            ]

            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = collections.deque([a.copy() for a in combined])
            self.k = 0

    def pop_next_action(self) -> Optional[np.ndarray]:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None

            act = np.asarray(self.cur_chunk.popleft(), dtype=np.float32)
            self.last_action = act.copy()
            self.k += 1
            return act

    def size(self) -> int:
        with self.lock:
            return len(self.cur_chunk)


class LongVLABridge(Node):
    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        control_fps: float,
        inference_rate: float,
        min_smooth_steps: int,
        latency_k: int,
        max_delta_per_step: float,
        smooth_alpha: float,
    ):
        super().__init__("longvla_bridge")

        self.bridge = CvBridge()
        self.prompt = prompt

        self.control_fps = float(control_fps)
        self.inference_rate = float(inference_rate)
        self.latency_k = int(latency_k)

        self.max_delta_per_step = float(max_delta_per_step)
        self.smooth_alpha = float(smooth_alpha)
        self.last_cmd: Optional[np.ndarray] = None

        self.lock = threading.Lock()
        self.infer_lock = threading.Lock()

        self.left_qpos = collections.deque(maxlen=FORCE_WINDOW)
        self.left_qvel = collections.deque(maxlen=FORCE_WINDOW)
        self.left_tau = collections.deque(maxlen=FORCE_WINDOW)

        self.right_qpos = collections.deque(maxlen=FORCE_WINDOW)
        self.right_qvel = collections.deque(maxlen=FORCE_WINDOW)
        self.right_tau = collections.deque(maxlen=FORCE_WINDOW)

        self.image_f: Optional[np.ndarray] = None
        self.image_l: Optional[np.ndarray] = None
        self.image_r: Optional[np.ndarray] = None

        self.stream_buffer = StreamActionBuffer(min_smooth_steps=min_smooth_steps)

        cb = ReentrantCallbackGroup()

        self.create_subscription(Image, "/camera_f/color/image_raw", self._image_f_cb, 5, callback_group=cb)
        self.create_subscription(Image, "/camera_l/color/image_raw", self._image_l_cb, 5, callback_group=cb)
        self.create_subscription(Image, "/camera_r/color/image_raw", self._image_r_cb, 5, callback_group=cb)

        self.create_subscription(JointState, "/left/joint_feedback_filtered", self._left_feedback_cb, 20, callback_group=cb)
        self.create_subscription(JointState, "/right/joint_feedback_filtered", self._right_feedback_cb, 20, callback_group=cb)

        # 这里必须是 Piper 真正订阅的 topic
        self.pub_left = self.create_publisher(JointState, "/left/joint_ctrl_cmd", 10)
        self.pub_right = self.create_publisher(JointState, "/right/joint_ctrl_cmd", 10)

        self.get_logger().info(f"Connecting to policy server at {host}:{port} ...")
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info("Connected.")

        self.control_timer = self.create_timer(1.0 / self.control_fps, self._control_step, callback_group=cb)
        self.infer_timer = self.create_timer(1.0 / self.inference_rate, self._infer_step, callback_group=cb)

        self.get_logger().info(
            f"Control loop @ {self.control_fps:.1f} Hz, inference @ {self.inference_rate:.1f} Hz."
        )

    # -------------------------
    # Image callbacks
    # -------------------------

    def _image_f_cb(self, msg: Image):
        with self.lock:
            self.image_f = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _image_l_cb(self, msg: Image):
        with self.lock:
            self.image_l = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _image_r_cb(self, msg: Image):
        with self.lock:
            self.image_r = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    # -------------------------
    # Joint callbacks
    # -------------------------

    def _left_feedback_cb(self, msg: JointState):
        if len(msg.position) < PER_ARM_JOINTS:
            return

        qpos = np.asarray(msg.position[:PER_ARM_JOINTS], dtype=np.float32)

        if len(msg.velocity) >= PER_ARM_JOINTS:
            qvel = np.asarray(msg.velocity[:PER_ARM_JOINTS], dtype=np.float32)
        else:
            qvel = np.zeros(PER_ARM_JOINTS, dtype=np.float32)

        if len(msg.effort) >= PER_ARM_JOINTS:
            tau = np.asarray(msg.effort[:PER_ARM_JOINTS], dtype=np.float32)
        else:
            tau = np.zeros(PER_ARM_JOINTS, dtype=np.float32)

        with self.lock:
            self.left_qpos.append(qpos)
            self.left_qvel.append(qvel)
            self.left_tau.append(tau)

    def _right_feedback_cb(self, msg: JointState):
        if len(msg.position) < PER_ARM_JOINTS:
            return

        qpos = np.asarray(msg.position[:PER_ARM_JOINTS], dtype=np.float32)

        if len(msg.velocity) >= PER_ARM_JOINTS:
            qvel = np.asarray(msg.velocity[:PER_ARM_JOINTS], dtype=np.float32)
        else:
            qvel = np.zeros(PER_ARM_JOINTS, dtype=np.float32)

        if len(msg.effort) >= PER_ARM_JOINTS:
            tau = np.asarray(msg.effort[:PER_ARM_JOINTS], dtype=np.float32)
        else:
            tau = np.zeros(PER_ARM_JOINTS, dtype=np.float32)

        with self.lock:
            self.right_qpos.append(qpos)
            self.right_qvel.append(qvel)
            self.right_tau.append(tau)

    # -------------------------
    # Observation
    # -------------------------

    def _build_observation(self) -> Optional[dict]:
        with self.lock:
            ready = (
                len(self.left_qpos) == FORCE_WINDOW
                and len(self.right_qpos) == FORCE_WINDOW
                and len(self.left_qvel) == FORCE_WINDOW
                and len(self.right_qvel) == FORCE_WINDOW
                and len(self.left_tau) == FORCE_WINDOW
                and len(self.right_tau) == FORCE_WINDOW
                and self.image_f is not None
                and self.image_l is not None
                and self.image_r is not None
            )

            if not ready:
                return None

            left_qpos = np.stack(list(self.left_qpos), axis=0).astype(np.float32)
            right_qpos = np.stack(list(self.right_qpos), axis=0).astype(np.float32)

            left_qvel = np.stack(list(self.left_qvel), axis=0).astype(np.float32)
            right_qvel = np.stack(list(self.right_qvel), axis=0).astype(np.float32)

            left_tau = np.stack(list(self.left_tau), axis=0).astype(np.float32)
            right_tau = np.stack(list(self.right_tau), axis=0).astype(np.float32)

            qpos_win = np.concatenate([left_qpos, right_qpos], axis=-1).astype(np.float32)
            qvel_win = np.concatenate([left_qvel, right_qvel], axis=-1).astype(np.float32)
            tau_win = np.concatenate([left_tau, right_tau], axis=-1).astype(np.float32)

            left_state = left_qpos[-1].astype(np.float32)
            right_state = right_qpos[-1].astype(np.float32)

            image_f = self.image_f.copy()
            image_l = self.image_l.copy()
            image_r = self.image_r.copy()

        return {
            "observation/qpos": qpos_win,
            "observation/qvel": qvel_win,
            "observation/filtered_effort": tau_win,
            "observation/left_state": left_state,
            "observation/right_state": right_state,
            "observation/image": image_f,
            "observation/left_image": image_l,
            "observation/right_image": image_r,
            "task": self.prompt,
            "prompt": self.prompt,
        }

    # -------------------------
    # Inference
    # -------------------------

    def _infer_step(self):
        if not self.infer_lock.acquire(blocking=False):
            return

        try:
            obs = self._build_observation()
            if obs is None:
                return

            result = self.client.infer(obs)

            if "actions" not in result:
                self.get_logger().error(f"policy result has no 'actions'. keys={list(result.keys())}")
                return

            actions = np.asarray(result["actions"], dtype=np.float32)

            if actions.ndim == 1:
                actions = actions[None, :]

            if actions.shape[-1] != NUM_JOINTS_TOTAL:
                self.get_logger().error(
                    f"Action dim mismatch: got {actions.shape}, expected last dim {NUM_JOINTS_TOTAL}"
                )
                return

            self.stream_buffer.integrate_new_chunk(actions, max_k=self.latency_k)
            self.get_logger().info(
                f"Integrated action chunk: shape={actions.shape}, buffer={self.stream_buffer.size()}"
            )

        except Exception as exc:
            self.get_logger().error(f"policy.infer 失败: {exc}")

        finally:
            self.infer_lock.release()

    # -------------------------
    # Publish
    # -------------------------

    def _smooth_and_limit_action(self, action_14: np.ndarray) -> np.ndarray:
        action_14 = np.asarray(action_14, dtype=np.float32)

        if self.last_cmd is None:
            self.last_cmd = action_14.copy()
            return action_14

        delta = np.clip(
            action_14 - self.last_cmd,
            -self.max_delta_per_step,
            self.max_delta_per_step,
        )
        limited = self.last_cmd + delta

        smoothed = self.smooth_alpha * limited + (1.0 - self.smooth_alpha) * self.last_cmd

        self.last_cmd = smoothed.copy()
        return smoothed

    def _publish_action(self, action_14: np.ndarray):
        action_14 = self._smooth_and_limit_action(action_14)

        left_action = action_14[:PER_ARM_JOINTS].copy()
        right_action = action_14[PER_ARM_JOINTS:].copy()

        stamp = self.get_clock().now().to_msg()

        msg_l = JointState()
        msg_l.header.stamp = stamp
        msg_l.name = ROS_JOINT_NAMES
        msg_l.position = [float(x) for x in left_action]
        msg_l.velocity = [DEFAULT_VELOCITY_LIMIT] * PER_ARM_JOINTS
        msg_l.effort = [0.0] * PER_ARM_JOINTS
        self.pub_left.publish(msg_l)

        msg_r = JointState()
        msg_r.header.stamp = stamp
        msg_r.name = ROS_JOINT_NAMES
        msg_r.position = [float(x) for x in right_action]
        msg_r.velocity = [DEFAULT_VELOCITY_LIMIT] * PER_ARM_JOINTS
        msg_r.effort = [0.0] * PER_ARM_JOINTS
        self.pub_right.publish(msg_r)

    def _control_step(self):
        action = self.stream_buffer.pop_next_action()
        if action is None:
            return

        self._publish_action(action)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="hand the bottle of tea to me")

    parser.add_argument("--control-fps", type=float, default=20.0)
    parser.add_argument("--inference-rate", type=float, default=3.0)

    parser.add_argument("--min-smooth-steps", type=int, default=8)
    parser.add_argument("--latency-k", type=int, default=0)

    parser.add_argument("--max-delta-per-step", type=float, default=0.01)
    parser.add_argument("--smooth-alpha", type=float, default=0.2)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rclpy.init()

    node = LongVLABridge(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        control_fps=args.control_fps,
        inference_rate=args.inference_rate,
        min_smooth_steps=args.min_smooth_steps,
        latency_k=args.latency_k,
        max_delta_per_step=args.max_delta_per_step,
        smooth_alpha=args.smooth_alpha,
    )

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()