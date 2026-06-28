#!/usr/bin/env python3
"""LongVLA RTC (Real-Time Chunking) deployment client.

三点相对 longvla_compressed.py 的升级:

1. **启动时机械臂强制复位**: 控制 timer 启动前, 主动下发 "all arm joints = 0,
   gripper open" 的目标, 保持 N 秒让 PID 收敛. 修复 "第一次启动 OK, 后续重启
   客户端成功率下降" 的失败模式——前一次任务结束时的 qpos / gripper 位置不一定
   跟训练数据起始帧一致, force_history[:, :, -1, 0] (当前帧 qpos) 偏移会让
   ForcePrior 的"期望力"判断错位.

2. **RTC 时间集成 (temporal ensemble)**: 不再"用新 chunk 替换旧 chunk"
   (longvla_compressed 的做法), 而是维护最近 K 个 chunk 的滑窗, 每个 control
   step 对所有 active chunk 加权平均, 权重按 chunk age 指数衰减. 出处:
   Zhao et al. ACT 2023. 优点:
     - chunk 边界 jerk 被多 chunk 平均掉, 不需要显式 overlap blending
     - 单次推理 outlier 被稀释, 推理服务有偶发噪声也不会立刻爆出去
     - inference rate 越高、active chunk 越多, 平滑度越好——这正是 RTC 哲学

3. **推理 latency 补偿 (`--chunk-start-offset`)**: 跳过新 chunk 头部已经过期的
   N 帧. 模型预测 chunk[k] 是为 obs 时刻 t0 后第 k+1 帧的目标, 但 client 发请求
   到收到 chunk 之间 robot 已经动了 ~L 个 control step. 不跳过 → ensemble
   里"新 chunk 的过期预测"和"旧 chunk 的当前预测"周期性互拉 → 移动过程中持续
   晃 (低频抖动). 这是 longvla_compressed 里 --latency-k 同样的修法.
"""
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


# ------------------------------------------------------------------------------
# 常量 — 与 longvla_compressed.py / 训练时 LongVLA conversion 保持一致
# ------------------------------------------------------------------------------
PER_ARM_JOINTS = 7
NUM_JOINTS_TOTAL = 14

FORCE_WINDOW = 16
TARGET_FPS = 30.0

JOINT_BUFFER_MAXLEN = 256
IMAGE_BUFFER_MAXLEN = 16

ROS_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

DEFAULT_VELOCITY_LIMIT = 5.2

# Piper joint7 = 夹爪, 14 维布局 = [L_j1..L_j6, L_gripper, R_j1..R_j6, R_gripper]
GRIPPER_INDICES = (PER_ARM_JOINTS - 1, 2 * PER_ARM_JOINTS - 1)  # (6, 13)

# 夹爪默认 "open" 位置 (per project memory, longvla_handover_0527 数据集). 用作
# 复位时的 fallback —— 如果用户启用了 hysteresis 离散化, 会以用户传入的 val_above
# 为准, 这两个常量只在 hysteresis 没启用时兜底.
DEFAULT_GRIPPER_OPEN_L = 0.075
DEFAULT_GRIPPER_OPEN_R = 0.095


# ------------------------------------------------------------------------------
# 工具函数 — 跟 longvla_compressed.py 同源, 维持行为一致
# ------------------------------------------------------------------------------

def _stamp_sec(msg) -> float:
    """ROS2 stamp → 单调秒 (float64), 方便插值."""
    s = msg.header.stamp
    return float(s.sec) + float(s.nanosec) * 1e-9


def _interp_to_grid(grid_t: np.ndarray, src_t: np.ndarray, src_x: np.ndarray) -> np.ndarray:
    """把 (src_t, src_x) 线性插值到 grid_t. 多维独立 1D interp."""
    out = np.empty((len(grid_t), src_x.shape[1]), dtype=np.float32)
    for d in range(src_x.shape[1]):
        out[:, d] = np.interp(grid_t, src_t, src_x[:, d])
    return out


def _nearest_index(stamps: np.ndarray, t_target: float) -> int:
    """在已排序 stamps 中找最接近 t_target 的下标. 用于图像最近邻匹配."""
    idx = int(np.searchsorted(stamps, t_target))
    if idx == 0:
        return 0
    if idx >= len(stamps):
        return len(stamps) - 1
    return idx - 1 if (t_target - stamps[idx - 1]) < (stamps[idx] - t_target) else idx


# ------------------------------------------------------------------------------
# RTC Temporal Ensemble Buffer
# ------------------------------------------------------------------------------

class RTCEnsembleBuffer:
    """ACT 风格 temporal ensemble: 滑窗 K 个 chunk, 每 step 加权平均.

    数据结构: chunks 是一个 list of (issued_step, chunk_array). 新 chunk 来到
    时 append; 老 chunk 过期 (issued_step + H ≤ current_step, 即所有动作都用完了)
    时 prune.

    每次 control step 调用 `pop_next_action()`:
      1. 找所有还有 "本 step 对应动作" 的 chunk —— 即 0 ≤ step - issued < H
      2. 取每个 chunk 在 local_idx = step - issued 的动作
      3. 权重: w_i = exp(-local_idx / age_tau)
         - local_idx=0 (chunk 头部) → w=1.0
         - local_idx=age_tau → w≈0.37
         - local_idx=3*age_tau → w≈0.05 (实质死亡)
         - age_tau 推荐: H/4 到 H/2, 论文里常用 H/4
      4. 输出加权平均

    与 longvla_compressed 的 StreamActionBuffer 对比:
      - StreamActionBuffer: 每次新 chunk 来时, 把"新 chunk 头部"和"旧 chunk
        剩余"做一次性线性 blend, 然后用新 chunk 覆盖 buffer. **只在两个 chunk
        交界做平滑**, chunk 内部还是按序执行.
      - RTCEnsembleBuffer: chunk 永远是"集体投票"出来的, 没有"当前 chunk"概念,
        任何瞬间的输出都是多个 chunk 的混合. inference rate 越高混合越浓.

    单 chunk 极端情况 (推理太慢、其他 chunk 都过期了): 等价于 open-loop 跑这
    单个 chunk, 没有 jerk 但也没有 RTC 优势. 推理频率应该 ≥ control_fps / H
    才能保证至少有 2 个 chunk overlapping.
    """

    def __init__(self, action_horizon: int, age_tau: float):
        self.H = int(action_horizon)
        self.tau = float(age_tau)
        self.lock = threading.Lock()
        # list[(issued_step, chunk: np.ndarray shape (≤H, action_dim))]
        self.chunks: list[tuple[int, np.ndarray]] = []
        self.step: int = 0
        self.last_action: Optional[np.ndarray] = None

    def add_chunk(self, chunk: np.ndarray) -> None:
        """记入新 chunk. 调用方需保证 chunk 是 (H_or_less, action_dim)."""
        with self.lock:
            chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            self.chunks.append((self.step, chunk))
            # Prune dead chunks: issued + H ≤ current_step → 本 chunk 已无可用动作
            self.chunks = [(s, c) for (s, c) in self.chunks if self.step - s < self.H]

    def pop_next_action(self) -> Optional[np.ndarray]:
        """控制线程每 1/control_fps 秒调用一次."""
        with self.lock:
            actions = []
            weights = []
            for (issued, chunk) in self.chunks:
                local_idx = self.step - issued
                if 0 <= local_idx < min(self.H, len(chunk)):
                    actions.append(chunk[local_idx])
                    # 指数衰减权重: 新 chunk (local_idx 小) 权重高
                    weights.append(float(np.exp(-local_idx / self.tau)))
            self.step += 1
            if not actions:
                # buffer 干涸 → 维持上一动作 (hold). 训练完合理的 policy 短暂 hold
                # 是安全的; 永久 hold 说明 inference 卡了, 上层应当 log 告警.
                return self.last_action
            arr = np.stack(actions, axis=0)              # (n_active, action_dim)
            w = np.array(weights, dtype=np.float32)
            w /= w.sum()
            avg = (arr * w[:, None]).sum(axis=0).astype(np.float32)
            self.last_action = avg.copy()
            return avg

    def size(self) -> int:
        with self.lock:
            return len(self.chunks)

    def reset(self) -> None:
        """复位: 清空 chunks, step 归零, last_action 清空. 用于机械臂复位之后."""
        with self.lock:
            self.chunks = []
            self.step = 0
            self.last_action = None


# ------------------------------------------------------------------------------
# 主节点
# ------------------------------------------------------------------------------

class LongVLARTCBridge(Node):
    """LongVLA + RTC + 启动复位.

    跟 longvla_compressed.LongVLABridge 的 API 基本一致, 主要差异:
      - 用 RTCEnsembleBuffer 替换 StreamActionBuffer
      - __init__ 末尾调用 _reset_arm_to_home() 阻塞等待手臂就位
      - 移除 latency_k (RTC 自带 chunk age decay, 不需要单独延迟补偿)
      - 移除 min_smooth_steps (没有 overlap 概念了)
    """

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        control_fps: float,
        inference_rate: float,
        rtc_age_tau: float,
        action_horizon: int,
        reset_seconds: float,
        chunk_start_offset: int,
        max_delta_per_step: float,
        smooth_alpha: float,
        arm_smooth_alpha: Optional[float] = None,
        gripper_thr_low: Optional[str] = None,
        gripper_thr_high: Optional[str] = None,
        gripper_val_below: Optional[str] = None,
        gripper_val_above: Optional[str] = None,
        zero_force_input: bool = False,
        skip_reset: bool = False,
    ):
        super().__init__("longvla_rtc_bridge")

        self.bridge = CvBridge()
        self.prompt = prompt

        self.control_fps = float(control_fps)
        self.inference_rate = float(inference_rate)
        self.action_horizon = int(action_horizon)
        self.reset_seconds = float(reset_seconds)
        self.skip_reset = bool(skip_reset)
        # chunk_start_offset: 推理 latency 补偿. 模型预测 chunk[k] 是 obs 时刻 t0 之后的
        # 第 k+1 帧目标. 但从 client 发请求到收到 chunk 之间 robot 已经动了 ~L control
        # step (取决于网络 + 推理耗时). chunk[0..L-1] 都已经过期了, 直接喂进 ensemble
        # 会让旧 chunk 在 age 0 时拉回时间上"L 步前"的位置 → ensemble 平均后出现
        # 周期性回拉 → 移动过程中持续晃. 跳过前 offset 帧让 chunk[0] 对齐到"现在".
        # 默认 4 ≈ 130ms inference latency @ 30Hz; 远程 server 可能要 5-6.
        self.chunk_start_offset = int(chunk_start_offset)

        self.max_delta_per_step = float(max_delta_per_step)
        self.smooth_alpha = float(smooth_alpha)
        self.arm_smooth_alpha = float(arm_smooth_alpha) if arm_smooth_alpha is not None else float(smooth_alpha)

        # 夹爪迟滞配置 (跟 longvla_compressed 完全一致, 不动 API)
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

        self.zero_force_input = bool(zero_force_input)
        if self.zero_force_input:
            self.get_logger().warn("zero_force_input=True: tau_win 将被置零再发送给 policy server")

        self.last_cmd: Optional[np.ndarray] = None

        self.lock = threading.Lock()
        self.infer_lock = threading.Lock()

        # 时间戳缓冲 (与 longvla_compressed 一致)
        self.left_qpos = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.left_qvel = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.left_tau = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.right_qpos = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.right_qvel = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.right_tau = collections.deque(maxlen=JOINT_BUFFER_MAXLEN)
        self.image_f_buf = collections.deque(maxlen=IMAGE_BUFFER_MAXLEN)
        self.image_l_buf = collections.deque(maxlen=IMAGE_BUFFER_MAXLEN)
        self.image_r_buf = collections.deque(maxlen=IMAGE_BUFFER_MAXLEN)

        # RTC ensemble buffer 替换 StreamActionBuffer
        self.rtc = RTCEnsembleBuffer(action_horizon=self.action_horizon, age_tau=rtc_age_tau)

        cb = ReentrantCallbackGroup()

        # 订阅 (跟 longvla_compressed 一致)
        self.get_logger().info("Subscribing to raw image topics")
        self.create_subscription(CompressedImage, "/camera_f/color/image_raw/compressed",
                                 self._image_f_cb, 5, callback_group=cb)
        self.create_subscription(CompressedImage, "/camera_l/color/image_raw/compressed",
                                 self._image_l_cb, 5, callback_group=cb)
        self.create_subscription(CompressedImage, "/camera_r/color/image_raw/compressed",
                                 self._image_r_cb, 5, callback_group=cb)
        self.create_subscription(JointState, "/left/joint_feedback_filtered",
                                 self._left_feedback_cb, 20, callback_group=cb)
        self.create_subscription(JointState, "/right/joint_feedback_filtered",
                                 self._right_feedback_cb, 20, callback_group=cb)

        # 发布
        self.pub_left = self.create_publisher(JointState, "/left/joint_ctrl_cmd", 10)
        self.pub_right = self.create_publisher(JointState, "/right/joint_ctrl_cmd", 10)

        # 复位 (核心新逻辑) — 必须在 control_timer 启动前做, 否则 control_timer
        # 会从空 rtc buffer pop 出 None 然后跳过, 但同时 control 命令上下文会乱.
        if not self.skip_reset:
            self._reset_arm_to_home(timeout_s=self.reset_seconds)
        else:
            self.get_logger().warn("--skip-reset 启用, 跳过启动复位. "
                                   "前次 deploy 状态可能污染本次推理, 仅用于调试.")

        # 连接 policy server
        self.get_logger().info(f"Connecting to policy server at {host}:{port} ...")
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info("Connected.")

        # 控制 + 推理 timer (在复位完成后才启动)
        self.control_timer = self.create_timer(1.0 / self.control_fps, self._control_step, callback_group=cb)
        self.infer_timer = self.create_timer(1.0 / self.inference_rate, self._infer_step, callback_group=cb)

        self.get_logger().info(
            f"Control loop @ {self.control_fps:.1f} Hz, inference @ {self.inference_rate:.1f} Hz, "
            f"RTC age_tau={rtc_age_tau:.1f} steps, action_horizon={self.action_horizon}."
        )

    # --------------------------------------------------------------------------
    # 启动复位 (核心新逻辑)
    # --------------------------------------------------------------------------

    def _home_pose(self) -> np.ndarray:
        """构造 home pose 14 维向量: 所有臂关节 = 0, 夹爪 = 张开值."""
        home = np.zeros(NUM_JOINTS_TOTAL, dtype=np.float32)
        for i, g in enumerate(GRIPPER_INDICES):
            if self.gripper_hysteresis_enabled and self.gripper_val_above is not None:
                # 用户传了 val_above (夹爪张开值), 优先用这个保持与推理时一致
                home[g] = self.gripper_val_above[i]
            else:
                home[g] = DEFAULT_GRIPPER_OPEN_L if i == 0 else DEFAULT_GRIPPER_OPEN_R
        return home

    def _publish_raw_joint_target(self, target_14: np.ndarray) -> None:
        """直接下发关节目标 (绕过 _smooth_and_limit_action). 用于复位."""
        stamp = self.get_clock().now().to_msg()
        for side, slice_, pub in (
            ("L", slice(0, PER_ARM_JOINTS), self.pub_left),
            ("R", slice(PER_ARM_JOINTS, NUM_JOINTS_TOTAL), self.pub_right),
        ):
            msg = JointState()
            msg.header.stamp = stamp
            msg.name = ROS_JOINT_NAMES
            msg.position = [float(x) for x in target_14[slice_]]
            # 用 1/4 默认速度上限, 复位是慢动作, 不需要全速
            # msg.velocity = [DEFAULT_VELOCITY_LIMIT * 0.25] * PER_ARM_JOINTS
            msg.velocity = [DEFAULT_VELOCITY_LIMIT] * PER_ARM_JOINTS
            msg.effort = [0.0] * PER_ARM_JOINTS
            pub.publish(msg)

    def _reset_arm_to_home(self, timeout_s: float = 5.0) -> None:
        """启动复位: 持续 ~timeout_s 秒下发 home pose 目标, 让 Piper PID 收敛.

        ROS2 publish 不需要 spin 也能发出去 (DDS 直发), 所以这里在 __init__ 里
        直接 publish + sleep 是合法的. 副作用是订阅回调 (image / joint feedback)
        在复位期间不处理, 这无所谓——复位完后 timer 一启动, 回调会立刻开始累积
        observation 缓冲, 等够 FORCE_WINDOW 帧 (~533ms) 后第一次推理就 ready.
        """
        home = self._home_pose()
        self.get_logger().info(
            f"Resetting arms to home pose for {timeout_s:.1f}s: "
            f"L_arm=zeros, L_gripper={home[GRIPPER_INDICES[0]]:.3f}, "
            f"R_arm=zeros, R_gripper={home[GRIPPER_INDICES[1]]:.3f}"
        )

        n_pubs_per_sec = 10
        n_total = int(timeout_s * n_pubs_per_sec)
        for _ in range(n_total):
            if not rclpy.ok():
                self.get_logger().warn("rclpy not ok during reset, abort")
                return
            self._publish_raw_joint_target(home)
            time.sleep(1.0 / n_pubs_per_sec)

        # 初始化 last_cmd = home, 这样 EMA 从已知点开始, 不会在第一个推理 chunk
        # 里出现 "上一动作未知 → 直接采纳新动作" 的跳变.
        self.last_cmd = home.copy()
        for i, g in enumerate(GRIPPER_INDICES):
            self.last_gripper[g] = float(home[g])

        # 同时 reset RTC buffer (本来就是空的, 但确保 step 计数从 0 开始)
        self.rtc.reset()

        self.get_logger().info("Arm reset complete. last_cmd initialized to home pose.")

    # --------------------------------------------------------------------------
    # 图像 / 关节回调 (跟 longvla_compressed 完全一致)
    # --------------------------------------------------------------------------

    def _compressed_to_rgb(self, msg: CompressedImage) -> np.ndarray:
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _parse_per_gripper(arg, n: int, name: str):
        if arg is None:
            return None
        if isinstance(arg, (int, float)):
            return [float(arg)] * n
        parts = [p.strip() for p in str(arg).split(",") if p.strip() != ""]
        if not parts:
            return None
        try:
            vals = [float(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"{name} 解析失败, 需要逗号分隔的浮点数: {arg}") from e
        if len(vals) == 1:
            vals = vals * n
        if len(vals) != n:
            raise ValueError(f"{name} 长度 {len(vals)} 与夹爪数 {n} 不匹配 (传入: {arg})")
        return vals

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
        qvel = (np.asarray(msg.velocity[:PER_ARM_JOINTS], dtype=np.float32)
                if len(msg.velocity) >= PER_ARM_JOINTS
                else np.zeros(PER_ARM_JOINTS, dtype=np.float32))
        tau = (np.asarray(msg.effort[:PER_ARM_JOINTS], dtype=np.float32)
               if len(msg.effort) >= PER_ARM_JOINTS
               else np.zeros(PER_ARM_JOINTS, dtype=np.float32))
        t = _stamp_sec(msg)
        with self.lock:
            self.left_qpos.append((t, qpos))
            self.left_qvel.append((t, qvel))
            self.left_tau.append((t, tau))

    def _right_feedback_cb(self, msg: JointState):
        if len(msg.position) < PER_ARM_JOINTS:
            return
        qpos = np.asarray(msg.position[:PER_ARM_JOINTS], dtype=np.float32)
        qvel = (np.asarray(msg.velocity[:PER_ARM_JOINTS], dtype=np.float32)
                if len(msg.velocity) >= PER_ARM_JOINTS
                else np.zeros(PER_ARM_JOINTS, dtype=np.float32))
        tau = (np.asarray(msg.effort[:PER_ARM_JOINTS], dtype=np.float32)
               if len(msg.effort) >= PER_ARM_JOINTS
               else np.zeros(PER_ARM_JOINTS, dtype=np.float32))
        t = _stamp_sec(msg)
        with self.lock:
            self.right_qpos.append((t, qpos))
            self.right_qvel.append((t, qvel))
            self.right_tau.append((t, tau))

    # --------------------------------------------------------------------------
    # Observation 构建 (跟 longvla_compressed 完全一致)
    # --------------------------------------------------------------------------

    def _build_observation(self) -> Optional[dict]:
        with self.lock:
            if not self.image_f_buf or not self.image_l_buf or not self.image_r_buf:
                self.get_logger().info(
                    f"Buffer not ready (images): "
                    f"f={len(self.image_f_buf)}, l={len(self.image_l_buf)}, r={len(self.image_r_buf)}"
                )
                return None
            if not self.left_qpos or not self.right_qpos:
                self.get_logger().info(
                    f"Buffer not ready (joints): L={len(self.left_qpos)}, R={len(self.right_qpos)}"
                )
                return None

            t_anchor = self.image_f_buf[-1][0]
            grid_t = t_anchor + (np.arange(FORCE_WINDOW, dtype=np.float64)
                                 - (FORCE_WINDOW - 1)) / TARGET_FPS
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
            f"align t_anchor={t_anchor:.3f}  "
            f"cam_l_drift={dt_l*1000:.0f}ms  cam_r_drift={dt_r*1000:.0f}ms  "
            f"L_joint_age={L_age*1000:.0f}ms  R_joint_age={R_age*1000:.0f}ms"
        )
        if dt_l > 2.0 / TARGET_FPS or dt_r > 2.0 / TARGET_FPS:
            self.get_logger().warn(
                f"camera drift > 2 frames: cam_l={dt_l*1000:.0f}ms, cam_r={dt_r*1000:.0f}ms"
            )

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

    # --------------------------------------------------------------------------
    # 推理 / 控制循环
    # --------------------------------------------------------------------------

    def _infer_step(self):
        """推理循环: 调 policy server 拿 chunk → 推入 RTC ensemble buffer."""
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

            # 推理 latency 补偿: 跳过 chunk 头部的"已经过去"那几帧, 让 chunk[0] 对齐到
            # 当前时刻. 不做这一步, RTC ensemble 会出现"新 chunk 拉回过去 / 旧 chunk
            # 走向未来"的周期性时序错配 → 移动过程中持续晃. 详见 __init__ 里 self.chunk_start_offset 注释.
            offset = min(self.chunk_start_offset, max(0, len(actions) - 1))
            if offset > 0:
                actions = actions[offset:]

            self.rtc.add_chunk(actions)
            self.get_logger().info(
                f"RTC: added chunk shape={actions.shape} (offset={offset}), "
                f"active_chunks={self.rtc.size()}"
            )

        except Exception as exc:
            self.get_logger().error(f"policy.infer 失败: {exc}")
        finally:
            self.infer_lock.release()

    def _smooth_and_limit_action(self, action_14: np.ndarray) -> np.ndarray:
        """跟 longvla_compressed 一致的 EMA + max_delta + 夹爪迟滞.

        注: RTC ensemble 本身已经做了 chunk 级别的平滑, 这一层 EMA 主要是
        per-step 速度限制 (安全网) 和夹爪离散化. 把 arm_smooth_alpha 调高
        (例如 0.5-0.8) 可以让 RTC 输出更直接通过, 减少二次延迟.
        """
        action_14 = np.asarray(action_14, dtype=np.float32)

        if self.last_cmd is None:
            # 理论上不应进入此分支 —— _reset_arm_to_home 已经把 last_cmd 设为 home.
            # 但 --skip-reset 时会走到这里, 兜底逻辑跟 longvla_compressed 一致.
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
            self.arm_smooth_alpha * limited + (1.0 - self.arm_smooth_alpha) * self.last_cmd[arm_mask]
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
        action = self.rtc.pop_next_action()
        if action is None:
            return
        self._publish_action(action)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LongVLA RTC deployment client (arm reset on startup + temporal ensemble).",
    )

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="hand the bottle of tea to me")

    parser.add_argument("--control-fps", type=float, default=20.0)
    parser.add_argument("--inference-rate", type=float, default=5.0,
                        help="推荐 5-10 Hz: 越高 RTC ensemble 里 active chunk 越多, "
                             "平滑度越好. longvla_compressed 默认 3 Hz 是为了 single-pair "
                             "blend, RTC ensemble 需要更高的频率才能发挥优势.")

    parser.add_argument("--rtc-age-tau", type=float, default=6.0,
                        help="RTC 权重衰减时间常数 (单位: control step). "
                             "默认 6 ≈ 0.3s @ 20Hz control. 调小更反应灵敏, 调大更平滑.")
    parser.add_argument("--action-horizon", type=int, default=25,
                        help="LongVLA 训练时的 action_horizon, 必须跟训练 config 一致.")

    parser.add_argument("--reset-seconds", type=float, default=3.0,
                        help="启动复位等待时间. PID 收敛到 home 需要 ~2-3s.")
    parser.add_argument("--skip-reset", action="store_true",
                        help="跳过启动复位 (仅调试用; 会复现 '首次 OK / 后续 degrade' 的问题).")

    parser.add_argument("--chunk-start-offset", type=int, default=4,
                        help="跳过新 chunk 的前 N 帧, 补偿推理 latency. "
                             "模型预测 chunk[k] 是 obs 时刻后第 k+1 帧, 但从发请求到 "
                             "收到 chunk 之间 robot 已经动了 ~N 个 control step. "
                             "不跳过 → ensemble 出现周期性回拉 → 移动中持续晃. "
                             "默认 4 ≈ 130ms latency @ 30Hz; 远程 server 可能要 5-6. "
                             "调试: 抖动消失 = 这个值合适; 还抖且方向反转 = 调小到 3 试试.")

    parser.add_argument("--max-delta-per-step", type=float, default=0.05,
                        help="每 control step 允许的最大关节变化 (rad). 默认 0.05 "
                             "= 0.05 × control_fps rad/s 速度上限. 跟 longvla_compressed "
                             "已 work 的设置对齐. 太小 → 命令追不上 RTC 输出, 阶梯型抖动.")
    parser.add_argument("--smooth-alpha", type=float, default=0.5,
                        help="比 longvla_compressed 默认 0.2 调高, 因为 RTC 已经做了 "
                             "chunk 级平滑, 这层 EMA 不需要太重.")
    parser.add_argument("--arm-smooth-alpha", type=float, default=None)

    parser.add_argument("--gripper-thr-low", type=str, default=None,
                        help='e.g. "0.025,0.085" 或单值广播 "0.05"')
    parser.add_argument("--gripper-thr-high", type=str, default=None,
                        help='e.g. "0.05,0.09"')
    parser.add_argument("--gripper-val-below", type=str, default=None,
                        help='raw 低于 thr_low 时输出 (一般 = 物理闭合位), e.g. "0.0,0.067"')
    parser.add_argument("--gripper-val-above", type=str, default=None,
                        help='raw 高于 thr_high 时输出 (一般 = 物理张开位), e.g. "0.075,0.095"')

    parser.add_argument("--zero-force-input", action="store_true",
                        help="把 tau_win 整体置零再送给 policy server (诊断用)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rclpy.init()

    node = LongVLARTCBridge(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        control_fps=args.control_fps,
        inference_rate=args.inference_rate,
        rtc_age_tau=args.rtc_age_tau,
        action_horizon=args.action_horizon,
        reset_seconds=args.reset_seconds,
        chunk_start_offset=args.chunk_start_offset,
        max_delta_per_step=args.max_delta_per_step,
        smooth_alpha=args.smooth_alpha,
        arm_smooth_alpha=args.arm_smooth_alpha,
        gripper_thr_low=args.gripper_thr_low,
        gripper_thr_high=args.gripper_thr_high,
        gripper_val_below=args.gripper_val_below,
        gripper_val_above=args.gripper_val_above,
        zero_force_input=args.zero_force_input,
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
