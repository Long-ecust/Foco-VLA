#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import logging
import pathlib
import re
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from openpi_client import websocket_client_policy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState


PER_ARM_JOINTS = 7
NUM_JOINTS_TOTAL = 14
IMAGE_BUFFER_MAXLEN = 16
JOINT_BUFFER_MAXLEN = 64

ROS_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
DEFAULT_VELOCITY_LIMIT = 5.2
GRIPPER_INDICES = (6, 13)

# 默认夹爪张开值；如果 CLI 传入 gripper_val_above，则优先使用 CLI。
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

    def clear(self):
        with self.lock:
            self.cur_chunk.clear()
            self.k = 0
            self.last_action = None


class OpenPIAgilexBridge(Node):
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
        gripper_thr_low: Optional[str] = None,
        gripper_thr_high: Optional[str] = None,
        gripper_val_below: Optional[str] = None,
        gripper_val_above: Optional[str] = None,
        csv_log_dir: Optional[str] = None,
        csv_model_name: str = "pi05_vision",
        csv_run_name: Optional[str] = None,
        reset_seconds: float = 3.0,
        skip_reset: bool = False,
    ):
        super().__init__("openpi_agilex_bridge")

        self.prompt = prompt
        self.control_fps = float(control_fps)
        self.inference_rate = float(inference_rate)
        self.latency_k = int(latency_k)
        self.max_delta_per_step = float(max_delta_per_step)
        self.smooth_alpha = float(smooth_alpha)
        self.arm_smooth_alpha = (
            float(arm_smooth_alpha) if arm_smooth_alpha is not None else float(smooth_alpha)
        )

        self.reset_seconds = float(reset_seconds)
        self.skip_reset = bool(skip_reset)


        n_gripper = len(GRIPPER_INDICES)
        self.gripper_thr_low = self._parse_per_gripper(gripper_thr_low, n_gripper, "gripper_thr_low")
        self.gripper_thr_high = self._parse_per_gripper(gripper_thr_high, n_gripper, "gripper_thr_high")
        self.gripper_val_below = self._parse_per_gripper(gripper_val_below, n_gripper, "gripper_val_below")
        self.gripper_val_above = self._parse_per_gripper(gripper_val_above, n_gripper, "gripper_val_above")

        self.gripper_hysteresis_enabled = (
            self.gripper_thr_low is not None
            and self.gripper_thr_high is not None
            and self.gripper_val_below is not None
            and self.gripper_val_above is not None
            and all(lo < hi for lo, hi in zip(self.gripper_thr_low, self.gripper_thr_high))
        )

        if self.gripper_hysteresis_enabled:
            self.last_gripper = {
                idx: self.gripper_val_above[i] for i, idx in enumerate(GRIPPER_INDICES)
            }
        else:
            self.last_gripper = {idx: 0.0 for idx in GRIPPER_INDICES}

        self.last_cmd: Optional[np.ndarray] = None
        self.start_time_monotonic = time.monotonic()
        self.csv_file = None
        self.csv_writer = None
        self.csv_event_file = None
        self.csv_event_writer = None
        self.csv_path: Optional[pathlib.Path] = None
        self.csv_event_path: Optional[pathlib.Path] = None
        self.csv_model_name = csv_model_name
        self.gripper_state = {idx: "open" for idx in GRIPPER_INDICES}
        self.gripper_phase = {idx: "pre_grasp" for idx in GRIPPER_INDICES}
        self.last_tau_abs_sum = 0.0

        self.lock = threading.Lock()
        self.infer_lock = threading.Lock()

        self.left_qpos = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.right_qpos = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.right_tau = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)

        self.image_f_buf = collections.deque(maxlen=IMAGE_BUFFER_MAXLEN)
        self.image_l_buf = collections.deque(maxlen=IMAGE_BUFFER_MAXLEN)
        self.image_r_buf = collections.deque(maxlen=IMAGE_BUFFER_MAXLEN)

        self.stream_buffer = StreamActionBuffer(min_smooth_steps=min_smooth_steps)
        self._setup_csv_logger(csv_log_dir, csv_model_name, csv_run_name)

        cb = ReentrantCallbackGroup()

        self.get_logger().info("Subscribing to compressed image topics")
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

        self.control_timer = self.create_timer(1.0 / self.control_fps, self._control_step, callback_group=cb)
        self.infer_timer = self.create_timer(1.0 / self.inference_rate, self._infer_step, callback_group=cb)

        self.get_logger().info(
            f"OpenPI Agilex client started. control={self.control_fps:.1f}Hz, "
            f"inference={self.inference_rate:.1f}Hz"
        )

    @staticmethod
    def _slugify(value: str, max_len: int = 80) -> str:
        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9._-]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("._-")
        return value[:max_len] or "empty"

    def _setup_csv_logger(
        self,
        csv_log_dir: Optional[str],
        csv_model_name: str,
        csv_run_name: Optional[str],
    ) -> None:
        if csv_log_dir is None:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = self._slugify(csv_model_name)
        prompt_slug = self._slugify(self.prompt)
        run_slug = self._slugify(csv_run_name) if csv_run_name else timestamp
        run_dir = pathlib.Path(csv_log_dir).expanduser() / model_slug / prompt_slug / run_slug
        run_dir.mkdir(parents=True, exist_ok=True)

        action_fieldnames = [
            "wall_time",
            "elapsed_s",
            "prompt",
            "model_name",
            "raw_r_gripper",
            "pub_r_gripper",
            "fb_r_gripper",
            "r_tau_abs_sum",
            "r_tau_abs_delta",
            "r_gripper_tau",
            "r_open_count",
            "r_close_count",
            "r_release_hold_remaining_s",
            "r_state",
            "r_phase",
            "event",
            "buffer_size",
        ]
        for j in range(PER_ARM_JOINTS):
            action_fieldnames.append(f"r_tau_j{j + 1}")

        self.csv_path = run_dir / "gripper_actions.csv"
        self.csv_file = self.csv_path.open("w", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=action_fieldnames)
        self.csv_writer.writeheader()

        self.csv_event_path = run_dir / "gripper_events.csv"
        self.csv_event_file = self.csv_event_path.open("w", newline="")
        self.csv_event_writer = csv.DictWriter(
            self.csv_event_file,
            fieldnames=[
                "wall_time",
                "elapsed_s",
                "prompt",
                "model_name",
                "event",
                "arm",
                "state",
                "phase",
                "raw_gripper",
                "pub_gripper",
                "fb_gripper",
                "tau_abs_sum",
                "tau_abs_delta",
                "gripper_tau",
                "open_count",
                "close_count",
                "release_hold_remaining_s",
            ],
        )
        self.csv_event_writer.writeheader()
        self.csv_file.flush()
        self.csv_event_file.flush()
        self.get_logger().info(f"CSV gripper log enabled: {self.csv_path}, events={self.csv_event_path}")

    def close_csv_logger(self) -> None:
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        if self.csv_event_file is not None:
            self.csv_event_file.flush()
            self.csv_event_file.close()
            self.csv_event_file = None
            self.csv_event_writer = None

    def _latest_right_feedback(self) -> tuple[float, np.ndarray]:
        with self.lock:
            r_gripper = (
                float(self.right_qpos[-1][1][PER_ARM_JOINTS - 1])
                if self.right_qpos
                else float("nan")
            )
            r_tau = (
                self.right_tau[-1][1].copy()
                if self.right_tau
                else np.full(PER_ARM_JOINTS, np.nan, dtype=np.float32)
            )
        return r_gripper, r_tau

    def _classify_right_gripper_state(self, value: float) -> str:
        if (
            self.gripper_hysteresis_enabled
            and self.gripper_val_below is not None
            and self.gripper_val_above is not None
        ):
            midpoint = 0.5 * (float(self.gripper_val_below[1]) + float(self.gripper_val_above[1]))
            return "closed" if value <= midpoint else "open"
        return "unknown"

    def _update_right_gripper_phase(
        self,
        raw_action_14: np.ndarray,
        pub_action_14: np.ndarray,
        fb_r: float,
        r_tau: np.ndarray,
        r_tau_abs_delta: float,
        now: float,
    ) -> tuple[str, str, str]:
        events = []
        g = GRIPPER_INDICES[1]
        prev_state = self.gripper_state[g]
        state = self._classify_right_gripper_state(float(pub_action_14[g]))

        if state != "unknown" and state != prev_state:
            event = f"R_{'CLOSE' if state == 'closed' else 'OPEN'}"
            events.append(event)
            if state == "closed":
                self.gripper_phase[g] = "grasp_or_carry"
            elif state == "open" and prev_state == "closed":
                self.gripper_phase[g] = "release"
            self.gripper_state[g] = state

            if self.csv_event_writer is not None:
                self.csv_event_writer.writerow(
                    {
                        "wall_time": time.time(),
                        "elapsed_s": now - self.start_time_monotonic,
                        "prompt": self.prompt,
                        "model_name": self.csv_model_name,
                        "event": event,
                        "arm": "right",
                        "state": state,
                        "phase": self.gripper_phase[g],
                        "raw_gripper": float(raw_action_14[g]),
                        "pub_gripper": float(pub_action_14[g]),
                        "fb_gripper": fb_r,
                        "tau_abs_sum": float(np.nansum(np.abs(r_tau))),
                        "tau_abs_delta": r_tau_abs_delta,
                        "gripper_tau": float(r_tau[PER_ARM_JOINTS - 1]),
                        "open_count": 0,
                        "close_count": 0,
                        "release_hold_remaining_s": 0.0,
                    }
                )
                self.csv_event_file.flush()

        if self.gripper_phase[g] == "release":
            self.gripper_phase[g] = "post_release"
        return self.gripper_state[g], self.gripper_phase[g], ";".join(events)

    def _write_gripper_csv_row(self, raw_action_14: np.ndarray, pub_action_14: np.ndarray) -> None:
        if self.csv_writer is None:
            return

        now = time.monotonic()
        r_idx = GRIPPER_INDICES[1]
        fb_r, r_tau = self._latest_right_feedback()
        r_tau_abs_sum = float(np.nansum(np.abs(r_tau)))
        r_tau_abs_delta = r_tau_abs_sum - self.last_tau_abs_sum
        self.last_tau_abs_sum = r_tau_abs_sum
        r_state, r_phase, event = self._update_right_gripper_phase(
            raw_action_14, pub_action_14, fb_r, r_tau, r_tau_abs_delta, now
        )

        row = {
            "wall_time": time.time(),
            "elapsed_s": now - self.start_time_monotonic,
            "prompt": self.prompt,
            "model_name": self.csv_model_name,
            "raw_r_gripper": float(raw_action_14[r_idx]),
            "pub_r_gripper": float(pub_action_14[r_idx]),
            "fb_r_gripper": fb_r,
            "r_tau_abs_sum": r_tau_abs_sum,
            "r_tau_abs_delta": r_tau_abs_delta,
            "r_gripper_tau": float(r_tau[PER_ARM_JOINTS - 1]),
            "r_open_count": 0,
            "r_close_count": 0,
            "r_release_hold_remaining_s": 0.0,
            "r_state": r_state,
            "r_phase": r_phase,
            "event": event,
            "buffer_size": self.stream_buffer.size(),
        }
        for j in range(PER_ARM_JOINTS):
            row[f"r_tau_j{j + 1}"] = float(r_tau[j])

        self.csv_writer.writerow(row)
        self.csv_file.flush()

    @staticmethod
    def _parse_per_gripper(arg, n: int, name: str):
        if arg is None:
            return None
        if isinstance(arg, (int, float)):
            return [float(arg)] * n

        parts = [p.strip() for p in str(arg).split(",") if p.strip() != ""]
        if len(parts) == 0:
            return None

        try:
            vals = [float(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"{name} 解析失败，需要逗号分隔浮点数: {arg}") from e

        if len(vals) == 1:
            vals = vals * n

        if len(vals) != n:
            raise ValueError(f"{name} 长度 {len(vals)} 与夹爪数 {n} 不匹配: {arg}")

        return vals

    def _home_pose(self) -> np.ndarray:
        home = np.zeros(NUM_JOINTS_TOTAL, dtype=np.float32)

        for i, g in enumerate(GRIPPER_INDICES):
            if self.gripper_hysteresis_enabled and self.gripper_val_above is not None:
                home[g] = float(self.gripper_val_above[i])
            else:
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

        for i, g in enumerate(GRIPPER_INDICES):
            self.last_gripper[g] = float(home[g])

        # 清空流式动作缓存，避免 reset 期间残留 action。
        self.stream_buffer.clear()

        self.get_logger().info("Arm reset complete. last_cmd initialized to home pose.")

    def _compressed_to_rgb(self, msg: CompressedImage) -> np.ndarray:
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _image_f_cb(self, msg: CompressedImage):
        try:
            img = self._compressed_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode front image: {e}")
            return

        with self.lock:
            self.image_f_buf.append((_stamp_sec(msg), img))

    def _image_l_cb(self, msg: CompressedImage):
        try:
            img = self._compressed_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode left image: {e}")
            return

        with self.lock:
            self.image_l_buf.append((_stamp_sec(msg), img))

    def _image_r_cb(self, msg: CompressedImage):
        try:
            img = self._compressed_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode right image: {e}")
            return

        with self.lock:
            self.image_r_buf.append((_stamp_sec(msg), img))

    def _left_feedback_cb(self, msg: JointState):
        if len(msg.position) < PER_ARM_JOINTS:
            return
        qpos = np.asarray(msg.position[:PER_ARM_JOINTS], dtype=np.float32)
        with self.lock:
            self.left_qpos.append((_stamp_sec(msg), qpos))

    def _right_feedback_cb(self, msg: JointState):
        if len(msg.position) < PER_ARM_JOINTS:
            return
        qpos = np.asarray(msg.position[:PER_ARM_JOINTS], dtype=np.float32)
        if len(msg.effort) >= PER_ARM_JOINTS:
            tau = np.asarray(msg.effort[:PER_ARM_JOINTS], dtype=np.float32)
        else:
            tau = np.zeros(PER_ARM_JOINTS, dtype=np.float32)
        with self.lock:
            self.right_qpos.append((_stamp_sec(msg), qpos))
            self.right_tau.append((_stamp_sec(msg), tau))

    def _build_observation(self) -> Optional[dict]:
        with self.lock:
            if not self.image_f_buf or not self.image_l_buf or not self.image_r_buf:
                self.get_logger().info(
                    f"Buffer not ready images: "
                    f"f={len(self.image_f_buf)}, l={len(self.image_l_buf)}, r={len(self.image_r_buf)}"
                )
                return None

            if not self.left_qpos or not self.right_qpos:
                self.get_logger().info(
                    f"Buffer not ready joints: L={len(self.left_qpos)}, R={len(self.right_qpos)}"
                )
                return None

            t_anchor = self.image_f_buf[-1][0]
            image_f = self.image_f_buf[-1][1].copy()

            cam_l_list = list(self.image_l_buf)
            cam_r_list = list(self.image_r_buf)

            left_t, left_state = self.left_qpos[-1]
            right_t, right_state = self.right_qpos[-1]

        cam_l_stamps = np.array([t for t, _ in cam_l_list], dtype=np.float64)
        cam_r_stamps = np.array([t for t, _ in cam_r_list], dtype=np.float64)

        idx_l = _nearest_index(cam_l_stamps, t_anchor)
        idx_r = _nearest_index(cam_r_stamps, t_anchor)

        image_l = cam_l_list[idx_l][1].copy()
        image_r = cam_r_list[idx_r][1].copy()

        state_14 = np.concatenate([left_state, right_state], axis=0).astype(np.float32)

        cam_l_drift = abs(cam_l_stamps[idx_l] - t_anchor)
        cam_r_drift = abs(cam_r_stamps[idx_r] - t_anchor)
        left_age = t_anchor - left_t
        right_age = t_anchor - right_t

        self.get_logger().info(
            f"align t_anchor={t_anchor:.3f} "
            f"cam_l_drift={cam_l_drift * 1000:.0f}ms "
            f"cam_r_drift={cam_r_drift * 1000:.0f}ms "
            f"L_joint_age={left_age * 1000:.0f}ms "
            f"R_joint_age={right_age * 1000:.0f}ms"
        )

        return {
            "images": {
                "top_head": image_f,
                "left_hand": image_l,
                "right_hand": image_r,
            },
            "state": state_14,
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
                self.get_logger().error(f"policy result has no actions. keys={list(result.keys())}")
                return

            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]

            if actions.shape[-1] != NUM_JOINTS_TOTAL:
                self.get_logger().error(
                    f"Action dim mismatch: got {actions.shape}, expected last dim={NUM_JOINTS_TOTAL}"
                )
                return

            self.stream_buffer.integrate_new_chunk(actions, max_k=self.latency_k)

            self.get_logger().info(
                f"Integrated action chunk: shape={actions.shape}, buffer={self.stream_buffer.size()}"
            )

        except Exception as exc:
            self.get_logger().error(f"policy.infer failed: {exc}")

        finally:
            self.infer_lock.release()

    def _smooth_and_limit_action(self, action_14: np.ndarray) -> np.ndarray:
        action_14 = np.asarray(action_14, dtype=np.float32)

        if self.last_cmd is None:
            init_cmd = action_14.copy()

            if self.gripper_hysteresis_enabled:
                for i, g in enumerate(GRIPPER_INDICES):
                    raw = float(action_14[g])
                    if raw < self.gripper_thr_low[i]:
                        init_cmd[g] = self.gripper_val_below[i]
                    elif raw > self.gripper_thr_high[i]:
                        init_cmd[g] = self.gripper_val_above[i]
                    else:
                        init_cmd[g] = self.last_gripper[g]
                    self.last_gripper[g] = init_cmd[g]

            self.last_cmd = init_cmd.copy()
            return init_cmd

        out = action_14.copy()

        arm_mask = np.ones(NUM_JOINTS_TOTAL, dtype=bool)
        if self.gripper_hysteresis_enabled:
            for g in GRIPPER_INDICES:
                arm_mask[g] = False

        delta = np.clip(
            out[arm_mask] - self.last_cmd[arm_mask],
            -self.max_delta_per_step,
            self.max_delta_per_step,
        )
        limited = self.last_cmd[arm_mask] + delta

        out[arm_mask] = (
            self.arm_smooth_alpha * limited
            + (1.0 - self.arm_smooth_alpha) * self.last_cmd[arm_mask]
        )

        if self.gripper_hysteresis_enabled:
            for i, g in enumerate(GRIPPER_INDICES):
                raw = float(action_14[g])
                if raw < self.gripper_thr_low[i]:
                    out[g] = self.gripper_val_below[i]
                elif raw > self.gripper_thr_high[i]:
                    out[g] = self.gripper_val_above[i]
                else:
                    out[g] = self.last_gripper[g]
                self.last_gripper[g] = out[g]

        self.last_cmd = out.copy()
        return out

    def _publish_action(self, action_14: np.ndarray):
        raw_action_14 = np.asarray(action_14, dtype=np.float32).copy()
        action_14 = self._smooth_and_limit_action(action_14)

        left_action = action_14[:PER_ARM_JOINTS].copy()
        right_action = action_14[PER_ARM_JOINTS:].copy()
        self._write_gripper_csv_row(raw_action_14, action_14)

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

    parser.add_argument("--control-fps", type=float, default=30.0)
    parser.add_argument("--inference-rate", type=float, default=4.0)

    parser.add_argument("--min-smooth-steps", type=int, default=8)
    parser.add_argument("--latency-k", type=int, default=0)

    parser.add_argument("--max-delta-per-step", type=float, default=0.05)
    parser.add_argument("--smooth-alpha", type=float, default=1.0)
    parser.add_argument("--arm-smooth-alpha", type=float, default=0.6)

    parser.add_argument("--gripper-thr-low", type=str, default=None)
    parser.add_argument("--gripper-thr-high", type=str, default=None)
    parser.add_argument("--gripper-val-below", type=str, default=None)
    parser.add_argument("--gripper-val-above", type=str, default=None)
    parser.add_argument(
        "--csv-log-dir",
        type=str,
        default=None,
        help="保存右臂夹爪/力矩 CSV 的根目录。目录结构: root/model_name/prompt/run_name/。",
    )
    parser.add_argument(
        "--csv-model-name",
        type=str,
        default="pi05_vision",
        help="写入 CSV 并用于目录归档的模型名。",
    )
    parser.add_argument(
        "--csv-run-name",
        type=str,
        default=None,
        help="本次实验 run 名。默认使用当前时间戳。",
    )

    parser.add_argument("--reset-seconds", type=float, default=3.0)
    parser.add_argument("--skip-reset", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rclpy.init()

    node = OpenPIAgilexBridge(
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
        gripper_thr_low=args.gripper_thr_low,
        gripper_thr_high=args.gripper_thr_high,
        gripper_val_below=args.gripper_val_below,
        gripper_val_above=args.gripper_val_above,
        csv_log_dir=args.csv_log_dir,
        csv_model_name=args.csv_model_name,
        csv_run_name=args.csv_run_name,
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
        node.close_csv_logger()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
