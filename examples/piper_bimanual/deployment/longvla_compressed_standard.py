#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from openpi_client import websocket_client_policy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState


PER_ARM_JOINTS = 7
NUM_JOINTS_TOTAL = 14

FORCE_WINDOW = 16
TARGET_FPS = 30.0

JOINT_BUFFER_MAXLEN = 256
IMAGE_BUFFER_MAXLEN = 16

ROS_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

# Piper 下发 JointState.velocity 字段。这个值不是模型输出速度，而是底层位置控制的速度限制。
# 如果复位或运动太猛，可以改成 1.5 或 1.0。
DEFAULT_VELOCITY_LIMIT = 5.2

# Piper 的 joint7 是夹爪。
GRIPPER_INDICES = (PER_ARM_JOINTS - 1, 2 * PER_ARM_JOINTS - 1)

# 标准评测版仅用于 reset home pose；推理时夹爪直接执行模型输出。
DEFAULT_GRIPPER_OPEN_L = 0.075
DEFAULT_GRIPPER_OPEN_R = 0.095

# 下发的夹爪夹持力矩（N·m），写入 JointState.effort 的夹爪维度（index PER_ARM_JOINTS-1）。
# piper_ros 节点会把它 clip 到 [0.5, 3.0] N·m 再换算成 SDK 单位（0.001 N·m）。
# 之前这里发 0.0，会被抬到 0.5（最小值）导致夹不紧；3.0 是节点当前允许的上限。
GRIPPER_EFFORT_NM = 1.5


def _gripper_effort_vec() -> list:
    """返回单臂 effort 列表：仅夹爪维度设为 GRIPPER_EFFORT_NM，其余关节为 0（节点忽略臂关节 effort）。"""
    eff = [0.0] * PER_ARM_JOINTS
    eff[PER_ARM_JOINTS - 1] = GRIPPER_EFFORT_NM
    return eff


def _stamp_sec(msg) -> float:
    s = msg.header.stamp
    return float(s.sec) + float(s.nanosec) * 1e-9


def _interp_to_grid(grid_t: np.ndarray, src_t: np.ndarray, src_x: np.ndarray) -> np.ndarray:
    out = np.empty((len(grid_t), src_x.shape[1]), dtype=np.float32)
    for d in range(src_x.shape[1]):
        out[:, d] = np.interp(grid_t, src_t, src_x[:, d])
    return out


def _nearest_index(stamps: np.ndarray, t_target: float) -> int:
    idx = int(np.searchsorted(stamps, t_target))
    if idx == 0:
        return 0
    if idx >= len(stamps):
        return len(stamps) - 1
    return idx - 1 if (t_target - stamps[idx - 1]) < (stamps[idx] - t_target) else idx


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

            # 延迟补偿：丢掉新 chunk 前几帧。
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

    def clear(self):
        with self.lock:
            self.cur_chunk.clear()
            self.k = 0
            self.last_action = None

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
        arm_smooth_alpha: Optional[float] = None,
        log_gripper_actions: bool = False,
        log_gripper_interval: float = 0.5,
        arm_deadband: float = 0.0,
        zero_force_input: bool = False,
        reset_seconds: float = 3.0,
        skip_reset: bool = False,
    ):
        super().__init__("longvla_bridge")

        self.bridge = CvBridge()
        self.prompt = prompt

        self.control_fps = float(control_fps)
        self.inference_rate = float(inference_rate)
        self.latency_k = int(latency_k)

        self.max_delta_per_step = float(max_delta_per_step)
        self.smooth_alpha = float(smooth_alpha)
        self.arm_smooth_alpha = (
            float(arm_smooth_alpha) if arm_smooth_alpha is not None else float(smooth_alpha)
        )
        self.log_gripper_actions = bool(log_gripper_actions)
        self.log_gripper_interval = max(0.0, float(log_gripper_interval))
        self._last_gripper_log_t = 0.0

        self.reset_seconds = float(reset_seconds)
        self.skip_reset = bool(skip_reset)

        # 机械臂死区（rad）：|目标 - 上一指令| 小于该值时不动，消除保持姿态时的微抖。0.0 = 关闭。
        self.arm_deadband = float(arm_deadband)

        self.zero_force_input = bool(zero_force_input)
        if self.zero_force_input:
            self.get_logger().warn("zero_force_input=True: tau_win 将被置零再发送给 policy server")

        self.last_cmd: Optional[np.ndarray] = None

        self.lock = threading.Lock()
        self.infer_lock = threading.Lock()

        self.left_qpos = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.left_qvel = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.left_tau = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)

        self.right_qpos = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.right_qvel = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.right_tau = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)

        self.image_f_buf = collections.deque(maxlen=IMAGE_BUFFER_MAXLEN)
        self.image_l_buf = collections.deque(maxlen=IMAGE_BUFFER_MAXLEN)
        self.image_r_buf = collections.deque(maxlen=IMAGE_BUFFER_MAXLEN)

        self.stream_buffer = StreamActionBuffer(min_smooth_steps=min_smooth_steps)

        cb = ReentrantCallbackGroup()

        self.get_logger().info("Subscribing to raw image topics")
        self.create_subscription(
            CompressedImage,
            "/camera_f/color/image_raw/compressed",
            self._image_f_cb,
            5,
            callback_group=cb,
        )
        self.create_subscription(
            CompressedImage,
            "/camera_l/color/image_raw/compressed",
            self._image_l_cb,
            5,
            callback_group=cb,
        )
        self.create_subscription(
            CompressedImage,
            "/camera_r/color/image_raw/compressed",
            self._image_r_cb,
            5,
            callback_group=cb,
        )

        self.create_subscription(
            JointState,
            "/left/joint_feedback_filtered",
            self._left_feedback_cb,
            20,
            callback_group=cb,
        )
        self.create_subscription(
            JointState,
            "/right/joint_feedback_filtered",
            self._right_feedback_cb,
            20,
            callback_group=cb,
        )

        self.pub_left = self.create_publisher(JointState, "/left/joint_ctrl_cmd", 10)
        self.pub_right = self.create_publisher(JointState, "/right/joint_ctrl_cmd", 10)

        # 先复位，再启动推理和控制 timer。
        # 这样不会出现 reset 和 policy action 同时抢控制的问题。
        if not self.skip_reset:
            self._reset_arm_to_home(timeout_s=self.reset_seconds)
        else:
            self.get_logger().warn(
                "--skip-reset enabled: 跳过启动复位，当前机械臂状态会直接进入 policy 推理。"
            )

        self.get_logger().info(f"Connecting to policy server at {host}:{port} ...")
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info("Connected.")

        self.control_timer = self.create_timer(
            1.0 / self.control_fps,
            self._control_step,
            callback_group=cb,
        )
        self.infer_timer = self.create_timer(
            1.0 / self.inference_rate,
            self._infer_step,
            callback_group=cb,
        )

        self.get_logger().info(
            f"Control loop @ {self.control_fps:.1f} Hz, inference @ {self.inference_rate:.1f} Hz."
        )

    def _compressed_to_rgb(self, msg: CompressedImage) -> np.ndarray:
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _home_pose(self) -> np.ndarray:
        home = np.zeros(NUM_JOINTS_TOTAL, dtype=np.float32)

        for i, g in enumerate(GRIPPER_INDICES):
            home[g] = DEFAULT_GRIPPER_OPEN_L if i == 0 else DEFAULT_GRIPPER_OPEN_R

        return home

    def _publish_raw_joint_target(self, target_14: np.ndarray) -> None:
        target_14 = np.asarray(target_14, dtype=np.float32)

        stamp = self.get_clock().now().to_msg()

        msg_l = JointState()
        msg_l.header.stamp = stamp
        msg_l.name = ROS_JOINT_NAMES
        msg_l.position = [float(x) for x in target_14[:PER_ARM_JOINTS]]
        msg_l.velocity = [DEFAULT_VELOCITY_LIMIT] * PER_ARM_JOINTS
        msg_l.effort = _gripper_effort_vec()
        self.pub_left.publish(msg_l)

        msg_r = JointState()
        msg_r.header.stamp = stamp
        msg_r.name = ROS_JOINT_NAMES
        msg_r.position = [float(x) for x in target_14[PER_ARM_JOINTS:]]
        msg_r.velocity = [DEFAULT_VELOCITY_LIMIT] * PER_ARM_JOINTS
        msg_r.effort = _gripper_effort_vec()
        self.pub_right.publish(msg_r)

    def _reset_arm_to_home(self, timeout_s: float = 3.0) -> None:
        home = self._home_pose()

        self.get_logger().info(
            f"Resetting arms to home for {timeout_s:.1f}s: "
            f"L_gripper={home[GRIPPER_INDICES[0]]:.3f}, "
            f"R_gripper={home[GRIPPER_INDICES[1]]:.3f}"
        )

        publish_hz = 10.0
        n_total = max(1, int(timeout_s * publish_hz))

        for _ in range(n_total):
            if not rclpy.ok():
                self.get_logger().warn("rclpy not ok during reset, abort reset.")
                return
            self._publish_raw_joint_target(home)
            time.sleep(1.0 / publish_hz)

        # 复位完成后，把滤波器初始状态设置为 home。
        # 否则第一帧 policy action 可能会从 None 直接跳到模型输出。
        self.last_cmd = home.copy()

        # 清空流式动作缓存，避免 reset 期间残留 action。
        self.stream_buffer.clear()

        self.get_logger().info("Arm reset complete. last_cmd initialized to home pose.")

    def _image_f_cb(self, msg: CompressedImage):
        try:
            img = self._compressed_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode camera_f image: {e}")
            return

        t = _stamp_sec(msg)
        with self.lock:
            self.image_f_buf.append((t, img))

    def _image_l_cb(self, msg: CompressedImage):
        try:
            img = self._compressed_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode camera_l image: {e}")
            return

        t = _stamp_sec(msg)
        with self.lock:
            self.image_l_buf.append((t, img))

    def _image_r_cb(self, msg: CompressedImage):
        try:
            img = self._compressed_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode camera_r image: {e}")
            return

        t = _stamp_sec(msg)
        with self.lock:
            self.image_r_buf.append((t, img))

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

        t = _stamp_sec(msg)

        with self.lock:
            self.left_qpos.append((t, qpos))
            self.left_qvel.append((t, qvel))
            self.left_tau.append((t, tau))

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

        t = _stamp_sec(msg)

        with self.lock:
            self.right_qpos.append((t, qpos))
            self.right_qvel.append((t, qvel))
            self.right_tau.append((t, tau))

    def _build_observation(self) -> Optional[dict]:
        with self.lock:
            if not self.image_f_buf or not self.image_l_buf or not self.image_r_buf:
                self.get_logger().info(
                    f"Buffer not ready images: "
                    f"f={len(self.image_f_buf)}, "
                    f"l={len(self.image_l_buf)}, "
                    f"r={len(self.image_r_buf)}"
                )
                return None

            if not self.left_qpos or not self.right_qpos:
                self.get_logger().info(
                    f"Buffer not ready joints: L={len(self.left_qpos)}, R={len(self.right_qpos)}"
                )
                return None

            t_anchor = self.image_f_buf[-1][0]

            grid_t = t_anchor + (
                np.arange(FORCE_WINDOW, dtype=np.float64) - (FORCE_WINDOW - 1)
            ) / TARGET_FPS

            t_oldest = float(grid_t[0])

            L_t = np.array([t for t, _ in self.left_qpos], dtype=np.float64)
            L_qpos = np.stack([v for _, v in self.left_qpos], axis=0).astype(np.float32)
            L_qvel = np.stack([v for _, v in self.left_qvel], axis=0).astype(np.float32)
            L_tau = np.stack([v for _, v in self.left_tau], axis=0).astype(np.float32)

            R_t = np.array([t for t, _ in self.right_qpos], dtype=np.float64)
            R_qpos = np.stack([v for _, v in self.right_qpos], axis=0).astype(np.float32)
            R_qvel = np.stack([v for _, v in self.right_qvel], axis=0).astype(np.float32)
            R_tau = np.stack([v for _, v in self.right_tau], axis=0).astype(np.float32)

            cam_l_list = list(self.image_l_buf)
            cam_r_list = list(self.image_r_buf)

            image_f = self.image_f_buf[-1][1].copy()

        if L_t[0] > t_oldest or R_t[0] > t_oldest:
            self.get_logger().info(
                f"Joint buffer doesn't cover window: need t<={t_oldest:.3f}, "
                f"L_oldest={L_t[0]:.3f}, R_oldest={R_t[0]:.3f}"
            )
            return None

        L_qpos_grid = _interp_to_grid(grid_t, L_t, L_qpos)
        L_qvel_grid = _interp_to_grid(grid_t, L_t, L_qvel)
        L_tau_grid = _interp_to_grid(grid_t, L_t, L_tau)

        R_qpos_grid = _interp_to_grid(grid_t, R_t, R_qpos)
        R_qvel_grid = _interp_to_grid(grid_t, R_t, R_qvel)
        R_tau_grid = _interp_to_grid(grid_t, R_t, R_tau)

        qpos_win = np.concatenate([L_qpos_grid, R_qpos_grid], axis=-1).astype(np.float32)
        qvel_win = np.concatenate([L_qvel_grid, R_qvel_grid], axis=-1).astype(np.float32)
        tau_win = np.concatenate([L_tau_grid, R_tau_grid], axis=-1).astype(np.float32)

        if self.zero_force_input:
            tau_win = np.zeros_like(tau_win)

        left_state = L_qpos_grid[-1].astype(np.float32)
        right_state = R_qpos_grid[-1].astype(np.float32)

        cam_l_stamps = np.array([t for t, _ in cam_l_list], dtype=np.float64)
        cam_r_stamps = np.array([t for t, _ in cam_r_list], dtype=np.float64)

        idx_l = _nearest_index(cam_l_stamps, t_anchor)
        idx_r = _nearest_index(cam_r_stamps, t_anchor)

        image_l = cam_l_list[idx_l][1].copy()
        image_r = cam_r_list[idx_r][1].copy()

        dt_l = abs(cam_l_stamps[idx_l] - t_anchor)
        dt_r = abs(cam_r_stamps[idx_r] - t_anchor)

        L_age = t_anchor - L_t[-1]
        R_age = t_anchor - R_t[-1]

        self.get_logger().info(
            f"align t_anchor={t_anchor:.3f} "
            f"cam_l_drift={dt_l * 1000:.0f}ms "
            f"cam_r_drift={dt_r * 1000:.0f}ms "
            f"L_joint_age={L_age * 1000:.0f}ms "
            f"R_joint_age={R_age * 1000:.0f}ms"
        )

        if dt_l > 2.0 / TARGET_FPS or dt_r > 2.0 / TARGET_FPS:
            self.get_logger().warn(
                f"camera drift > 2 frames: cam_l={dt_l * 1000:.0f}ms, "
                f"cam_r={dt_r * 1000:.0f}ms"
            )

        # 这个限制建议保留。joint feedback 太旧时，模型会基于错误状态推理，容易出现回退/修正。
        if L_age > 0.15 or R_age > 0.15:
            self.get_logger().warn(
                f"stale joint feedback: L_age={L_age * 1000:.0f}ms, "
                f"R_age={R_age * 1000:.0f}ms, skip inference"
            )
            return None

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
                    f"Action dim mismatch: got {actions.shape}, "
                    f"expected last dim {NUM_JOINTS_TOTAL}"
                )
                return

            self.stream_buffer.integrate_new_chunk(actions, max_k=self.latency_k)

            self.get_logger().info(
                f"Integrated action chunk: shape={actions.shape}, "
                f"buffer={self.stream_buffer.size()}"
            )

        except Exception as exc:
            self.get_logger().error(f"policy.infer 失败: {exc}")

        finally:
            self.infer_lock.release()

    def _smooth_and_limit_action(self, action_14: np.ndarray) -> np.ndarray:
        action_14 = np.asarray(action_14, dtype=np.float32)

        if self.last_cmd is None:
            self.last_cmd = action_14.copy()
            return action_14

        out = action_14.copy()

        arm_mask = np.ones(NUM_JOINTS_TOTAL, dtype=bool)
        for g in GRIPPER_INDICES:
            arm_mask[g] = False

        raw_delta = out[arm_mask] - self.last_cmd[arm_mask]

        # 死区：把小于 arm_deadband 的目标位移归零，消除保持姿态时的微抖。
        if self.arm_deadband > 0.0:
            raw_delta = np.where(np.abs(raw_delta) < self.arm_deadband, 0.0, raw_delta)

        delta = np.clip(raw_delta, -self.max_delta_per_step, self.max_delta_per_step)

        limited = self.last_cmd[arm_mask] + delta

        out[arm_mask] = (
            self.arm_smooth_alpha * limited
            + (1.0 - self.arm_smooth_alpha) * self.last_cmd[arm_mask]
        )

        self.last_cmd = out.copy()
        return out

    def _publish_action(self, action_14: np.ndarray):
        raw_action_14 = np.asarray(action_14, dtype=np.float32).copy()
        action_14 = self._smooth_and_limit_action(action_14)

        left_action = action_14[:PER_ARM_JOINTS].copy()
        right_action = action_14[PER_ARM_JOINTS:].copy()

        if self.log_gripper_actions:
            now = time.monotonic()
            if now - self._last_gripper_log_t >= self.log_gripper_interval:
                self._last_gripper_log_t = now
                self.get_logger().info(
                    "gripper raw/publish: "
                    f"L_raw={raw_action_14[GRIPPER_INDICES[0]]:.4f}, "
                    f"R_raw={raw_action_14[GRIPPER_INDICES[1]]:.4f}, "
                    f"L_pub={action_14[GRIPPER_INDICES[0]]:.4f}, "
                    f"R_pub={action_14[GRIPPER_INDICES[1]]:.4f}"
                )

        stamp = self.get_clock().now().to_msg()

        msg_l = JointState()
        msg_l.header.stamp = stamp
        msg_l.name = ROS_JOINT_NAMES
        msg_l.position = [float(x) for x in left_action]
        msg_l.velocity = [DEFAULT_VELOCITY_LIMIT] * PER_ARM_JOINTS
        msg_l.effort = _gripper_effort_vec()
        self.pub_left.publish(msg_l)

        msg_r = JointState()
        msg_r.header.stamp = stamp
        msg_r.name = ROS_JOINT_NAMES
        msg_r.position = [float(x) for x in right_action]
        msg_r.velocity = [DEFAULT_VELOCITY_LIMIT] * PER_ARM_JOINTS
        msg_r.effort = _gripper_effort_vec()
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

    parser.add_argument("--reset-seconds", type=float, default=3.0)
    parser.add_argument("--skip-reset", action="store_true")

    parser.add_argument("--max-delta-per-step", type=float, default=0.01)
    parser.add_argument("--smooth-alpha", type=float, default=0.2)
    parser.add_argument("--arm-smooth-alpha", type=float, default=None)
    parser.add_argument(
        "--log-gripper-actions",
        action="store_true",
        help="打印模型 raw 夹爪目标和最终下发夹爪目标；只记录，不改变控制。",
    )
    parser.add_argument(
        "--log-gripper-interval",
        type=float,
        default=0.5,
        help="夹爪日志打印间隔，单位秒。默认 0.5s。",
    )

    parser.add_argument(
        "--arm-deadband",
        type=float,
        default=0.0,
        help="机械臂死区(rad)：|目标-上一指令| 小于该值时不动，消除保持姿态时的微抖。0=关闭，可试 0.005~0.008。",
    )

    parser.add_argument(
        "--zero-force-input",
        action="store_true",
        help="把 tau_win 整体置零再送给 policy server，诊断用。",
    )

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
        arm_smooth_alpha=args.arm_smooth_alpha,
        log_gripper_actions=args.log_gripper_actions,
        log_gripper_interval=args.log_gripper_interval,
        arm_deadband=args.arm_deadband,
        zero_force_input=args.zero_force_input,
        reset_seconds=args.reset_seconds,
        skip_reset=args.skip_reset,
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
