#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import logging
import threading
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


PER_ARM_JOINTS = 7  #每条机械臂7个关节
NUM_JOINTS_TOTAL = 14   #总动作维度: 左右臂各7个关节 共计14维动作输出

#Force window 长度: 训练时 conversion 把 qpos/qvel/tau 在 30Hz 网格上插值,
#再用 lerobot 的 delta_timestamps 取 16 个连续帧 = 533ms 历史窗口.
#这个FORCE_WINDOW是力矩历史窗口长度, 默认16帧
FORCE_WINDOW = 16
TARGET_FPS = 30.0  #网格采样频率是30HZ，与训练时保持一致

#时间戳缓冲区容量 (按各 topic 实际 Hz 估算上限留余量)
JOINT_BUFFER_MAXLEN = 256   #关节回调通常200Hz左右, 256 帧 ≈ 1.3-2.5s。能够覆盖16帧时动作的静态噪声。
IMAGE_BUFFER_MAXLEN = 16    #图像约 30Hz, 16 帧 ≈ 533ms; 仅用于最近邻匹配

ROS_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

DEFAULT_VELOCITY_LIMIT = 5.2    #Piper机械臂下发，默认速度是5.2m/s

# Piper 的 joint7 是夹爪开合, 因此夹爪在 14 维中的位置是 6 (左) 和 13 (右).
GRIPPER_INDICES = (PER_ARM_JOINTS - 1, 2 * PER_ARM_JOINTS - 1) #(6, 13)


#将ROS2的时间戳转换为单调秒数（float64），以便后续处理和对齐使用。
def _stamp_sec(msg) -> float:
    s = msg.header.stamp
    return float(s.sec) + float(s.nanosec) * 1e-9

#将源时间序列 (src_t, src_x) 线性插值到目标时间网格 grid_t。
#对每个维度独立做 1D 线性插值,超出范围的点用边界值外推
#grid_t是目标时间网格，src_t是源时间戳序列，src_x是对应的数值序列（每行对应一个时间戳，每列对应一个维度）。函数返回在 grid_t 上插值后的数值序列。
def _interp_to_grid(grid_t: np.ndarray, src_t: np.ndarray, src_x: np.ndarray) -> np.ndarray:
    out = np.empty((len(grid_t), src_x.shape[1]), dtype=np.float32)
    for d in range(src_x.shape[1]):
        #对第d维做1D线性插值
        out[:, d] = np.interp(grid_t, src_t, src_x[:, d])
    return out


def _nearest_index(stamps: np.ndarray, t_target: float) -> int:
    #在已排序 stamps 中找最接近 t_target 的下标.用于图像最近邻匹配（图像缓冲较浅，不做插值）。
    idx = int(np.searchsorted(stamps, t_target))    #二分查找
    if idx == 0:
        return 0
    if idx >= len(stamps):
        return len(stamps) - 1
    #比较t_target离哪个更近
    return idx - 1 if (t_target - stamps[idx - 1]) < (stamps[idx] - t_target) else idx

#流式动作缓冲与平滑器
#推理3hz（可改），控制30hz,每次推理有一个action序列，需要在新旧action序列的重叠部分做加权平滑，避免跳变
class StreamActionBuffer:
    def __init__(self, min_smooth_steps: int = 8):
        self.cur_chunk = collections.deque()    #当前action缓冲队列
        self.k = 0  #从缓冲中已取出的action计数
        self.last_action: Optional[np.ndarray] = None   #上一个时刻的action
        self.lock = threading.Lock()
        self.min_smooth_steps = int(min_smooth_steps)   #最少平滑帧数（新旧序列重叠不足时，用上一个时刻action填充到这个长度）

    #将新推理的 action 序列集成到缓冲中，处理与旧序列的平滑过渡。
    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int = 0):
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return

            actions_chunk = np.asarray(actions_chunk, dtype=np.float32)
            if actions_chunk.ndim == 1:
                actions_chunk = actions_chunk[None, :]  #将1维数组变为二维（14,）转化成（1, 14）
            #延迟补偿，丢弃掉最早的drop_n帧
            drop_n = min(self.k, int(max_k))
            if drop_n >= len(actions_chunk):
                #新chunk完全被延迟遮盖，本次集成无效，直接丢弃
                return

            new_list = [a.copy() for a in actions_chunk[drop_n:]]   #保留从drop_n开始的有效部分，转换成列表并复制
            old_list = list(self.cur_chunk)

            #如果缓存是空的话，用last_action补充到min_smooth_steps
            if len(old_list) == 0:
                if self.last_action is not None:
                    #重复上一时刻action用来初始化平滑缓冲
                    old_list = [self.last_action.copy() for _ in range(self.min_smooth_steps)]
                else:
                    #第一次推理，没有last_action，直接用新的chunk
                    self.cur_chunk = collections.deque(new_list)
                    self.k = 0
                    return

            #如果不足的话，将old_list填充到min_smooth_steps
            if len(old_list) < self.min_smooth_steps:
                old_list += [old_list[-1].copy() for _ in range(self.min_smooth_steps - len(old_list))]
            #计算重叠区间长度
            overlap_len = min(len(old_list), len(new_list))

            if overlap_len <= 0:
                #如果没有重叠的话，直接用新的chunk
                self.cur_chunk = collections.deque(new_list)
                self.k = 0
                return
            #重叠部分线型段进行加权平滑
            #w_old是从1到0的权重，w_new是从0到1的权重，长度为overlap_len
            w_old = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32)
            w_new = 1.0 - w_old

            smoothed = [
                w_old[i] * np.asarray(old_list[i], dtype=np.float32)
                + w_new[i] * np.asarray(new_list[i], dtype=np.float32)
                for i in range(overlap_len)
            ]
            #拼接平滑过渡+新的chunk里面剩余部分
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = collections.deque([a.copy() for a in combined])
            self.k = 0  #复位计数器

    #从缓冲中弹出下一个 action（控制线程调用）。
    def pop_next_action(self) -> Optional[np.ndarray]:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None

            act = np.asarray(self.cur_chunk.popleft(), dtype=np.float32)
            self.last_action = act.copy()
            self.k += 1  # 计数器递增
            return act

    #返回当前缓冲中的 action 数量。
    def size(self) -> int:
        with self.lock:
            return len(self.cur_chunk)


#LongVLA推理类
class LongVLABridge(Node):
    def __init__(
        self,
        host: str,  # 推理服务器地址
        port: int,  # 推理服务器端口
        prompt: str,  # 任务语义描述
        control_fps: float,  # 控制帧率
        inference_rate: float,  # 推理帧率
        min_smooth_steps: int,  # 最小平滑步数
        latency_k: int,  # 延迟补偿帧数
        max_delta_per_step: float,  # 每步最大变化量（用来限制速度）
        smooth_alpha: float,  # 手臂的EMA平滑系数，从0到1，
                            #0是完全保留上一时刻动作（最平滑但是延迟大），1是完全采用当前推理动作
        arm_smooth_alpha: Optional[float] = None,   #手臂通道单独的 smooth_alpha，不指定则用 smooth_alpha
        gripper_thr_low: Optional[str] = None,      #夹爪离散化下阈值
        gripper_thr_high: Optional[str] = None,     #夹爪离散化上阈值
        gripper_val_below: Optional[str] = None,    #夹爪 raw < thr_low 时的离散输出值（闭合）
        gripper_val_above: Optional[str] = None,    #夹爪 raw > thr_high 时的离散输出值（张开）
        zero_force_input: bool = False,     #诊断开关，推理时把力矩窗口置零
    ):
        super().__init__("longvla_bridge")

        self.bridge = CvBridge()
        self.prompt = prompt

        self.control_fps = float(control_fps)
        self.inference_rate = float(inference_rate)
        self.latency_k = int(latency_k)

        self.max_delta_per_step = float(max_delta_per_step)
        self.smooth_alpha = float(smooth_alpha)
        # 手臂通道单独的 EMA 系数; 未设置时退化到 smooth_alpha (向后兼容).
        self.arm_smooth_alpha = float(arm_smooth_alpha) if arm_smooth_alpha is not None else float(smooth_alpha)

        #   raw <  thr_low 输出 val_below （闭合）
        #   raw >  thr_high 输出val_above  （张开）
        #   thr_low ≤ raw ≤ thr_high → 保持上一时刻 (死区, 防抖动)
        # 每个参数都接受逗号分隔的 N 个值 (N = 夹爪数, 当前=2 即 L,R),
        # 这 4 个参数任一未设则不启用迟滞，夹爪走连续 EMA
        n_gripper = len(GRIPPER_INDICES)
        self.gripper_thr_low = self._parse_per_gripper(gripper_thr_low, n_gripper, "gripper_thr_low")
        self.gripper_thr_high = self._parse_per_gripper(gripper_thr_high, n_gripper, "gripper_thr_high")
        self.gripper_val_below = self._parse_per_gripper(gripper_val_below, n_gripper, "gripper_val_below")
        self.gripper_val_above = self._parse_per_gripper(gripper_val_above, n_gripper, "gripper_val_above")
        # 检查迟滞是否完整启用
        self.gripper_hysteresis_enabled = (
            self.gripper_thr_low is not None
            and self.gripper_thr_high is not None
            and self.gripper_val_below is not None
            and self.gripper_val_above is not None
            and all(lo < hi for lo, hi in zip(self.gripper_thr_low, self.gripper_thr_high))
        )
        # 初始默认夹爪状态置为 val_above (张开)
        if self.gripper_hysteresis_enabled:
            self.last_gripper = {
                idx: self.gripper_val_above[i] for i, idx in enumerate(GRIPPER_INDICES)
            }
        else:
            self.last_gripper = {idx: 0.0 for idx in GRIPPER_INDICES}

        # 诊断开关，用于力输入置零,对比实验要不要仅用视觉/仅用力来输入，观察力对模型的影响
        # 用来 A/B 测试 LongVLA 的 ForcePrior/ForceTokenizer 是否依赖力信号
        # 如果 zero_force_input=True，则 tau_win 整体置零再发送给 server
        self.zero_force_input = bool(zero_force_input)
        if self.zero_force_input:
            self.get_logger().warn("zero_force_input=True: tau_win 将被置零再发送给 policy server")

        self.last_cmd: Optional[np.ndarray] = None  #上一个时刻的命令（用于EMA）

        self.lock = threading.Lock()
        self.infer_lock = threading.Lock()

        # 时间戳缓冲
        # 每个 deque 存 (stamp_sec, value) tuple. value 为 np.ndarray (关节信号) 或 np.ndarray (图像).
        # 关节缓冲较深 (≥ FORCE_WINDOW/TARGET_FPS 秒 × 实际回调频率), 保证插值时能完整覆盖回溯窗口.
        # 图像缓冲较浅, 只为最近邻匹配前置相机时间戳.
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

        # 订阅原始图像话题
        self.get_logger().info("Subscribing to raw image topics")
        self.create_subscription(CompressedImage, "/camera_f/color/image_raw/compressed", self._image_f_cb, 5, callback_group=cb)
        self.create_subscription(CompressedImage, "/camera_l/color/image_raw/compressed", self._image_l_cb, 5, callback_group=cb)
        self.create_subscription(CompressedImage, "/camera_r/color/image_raw/compressed", self._image_r_cb, 5, callback_group=cb)
        # 订阅关节状态话题
        self.create_subscription(JointState, "/left/joint_feedback_filtered", self._left_feedback_cb, 20, callback_group=cb)
        self.create_subscription(JointState, "/right/joint_feedback_filtered", self._right_feedback_cb, 20, callback_group=cb)

        # 发布关节控制命令
        self.pub_left = self.create_publisher(JointState, "/left/joint_ctrl_cmd", 10)
        self.pub_right = self.create_publisher(JointState, "/right/joint_ctrl_cmd", 10)

        # 连接服务器
        self.get_logger().info(f"Connecting to policy server at {host}:{port} ...")
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info("Connected.")

        # 创建定时器：控制循环+推理循环
        self.control_timer = self.create_timer(1.0 / self.control_fps, self._control_step, callback_group=cb)
        self.infer_timer = self.create_timer(1.0 / self.inference_rate, self._infer_step, callback_group=cb)

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
    @staticmethod
    # 夹抓参数解析
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
            raise ValueError(f"{name} 解析失败, 需要逗号分隔的浮点数: {arg}") from e
        if len(vals) == 1:
            vals = vals * n
        if len(vals) != n:
            raise ValueError(f"{name} 长度 {len(vals)} 与夹爪数 {n} 不匹配 (传入: {arg})")
        return vals
    #图像回调函数
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

    # 关节状态回调函数，存储qpos、qvel、tau+时间辍
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
        # 构建 observation.
        # 系统参考时间辍取前相机最新的时间辍t_anchor
        # 回溯时间网络，从t_anchor开始往前 FORCE_WINDOW 帧，步长 1/TARGET_FPS, 共 FORCE_WINDOW 点
        # 关节信号插值，线型差值到grid_t
        # 图像最邻近匹配到 t_anchor
        # 组装 observation 字典

        with self.lock:
            # 基础就绪检查
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

            # 锚点 = 前置相机最新帧时间戳
            t_anchor = self.image_f_buf[-1][0]

            # 回溯网格，从t_anchor 开始往前 FORCE_WINDOW 帧，步长 1/TARGET_FPS, 共 FORCE_WINDOW 点
            # grid_t[0] = t_anchor - 15/30 (最旧的采样点)
            # grid_t[15] = t_anchor (最新的采样点)
            grid_t = t_anchor + (np.arange(FORCE_WINDOW, dtype=np.float64)
                                 - (FORCE_WINDOW - 1)) / TARGET_FPS
            t_oldest = float(grid_t[0])

            # 拷贝缓冲到局部数组（避免长持锁）
            # 这样可以在释放锁后再做耗时的numpy操作
            L_t = np.array([t for t, _ in self.left_qpos], dtype=np.float64)
            L_qpos = np.stack([v for _, v in self.left_qpos], axis=0).astype(np.float32)
            L_qvel = np.stack([v for _, v in self.left_qvel], axis=0).astype(np.float32)
            L_tau  = np.stack([v for _, v in self.left_tau], axis=0).astype(np.float32)

            R_t = np.array([t for t, _ in self.right_qpos], dtype=np.float64)
            R_qpos = np.stack([v for _, v in self.right_qpos], axis=0).astype(np.float32)
            R_qvel = np.stack([v for _, v in self.right_qvel], axis=0).astype(np.float32)
            R_tau  = np.stack([v for _, v in self.right_tau], axis=0).astype(np.float32)

            cam_l_list = list(self.image_l_buf)
            cam_r_list = list(self.image_r_buf)
            image_f = self.image_f_buf[-1][1].copy()

        # 检查: 关节最老时间必须 ≤ 网格起点, 否则数据不够回溯
        # 若然 Joints 最早时间 < 锚点, 则数据不够回溯
        if L_t[0] > t_oldest or R_t[0] > t_oldest:
            self.get_logger().info(
                f"Joint buffer doesn't cover window: need t<={t_oldest:.3f}, "
                f"L_oldest={L_t[0]:.3f}, R_oldest={R_t[0]:.3f}"
            )
            return None

        # 线性插值，插值到grid_t
        L_qpos_grid = _interp_to_grid(grid_t, L_t, L_qpos)  # shape (16, 7)
        L_qvel_grid = _interp_to_grid(grid_t, L_t, L_qvel)  # shape (16, 7)
        L_tau_grid  = _interp_to_grid(grid_t, L_t, L_tau)   # shape (16, 7)

        R_qpos_grid = _interp_to_grid(grid_t, R_t, R_qpos)  # shape (16, 7)
        R_qvel_grid = _interp_to_grid(grid_t, R_t, R_qvel)  # shape (16, 7)
        R_tau_grid  = _interp_to_grid(grid_t, R_t, R_tau)  # shape (16, 7)

        # 拼成 (FORCE_WINDOW, 14) 左7+右7
        qpos_win = np.concatenate([L_qpos_grid, R_qpos_grid], axis=-1).astype(np.float32)
        qvel_win = np.concatenate([L_qvel_grid, R_qvel_grid], axis=-1).astype(np.float32)
        tau_win  = np.concatenate([L_tau_grid,  R_tau_grid],  axis=-1).astype(np.float32)

        # 测试force设置为0的情况，观察力输入对模型的影响
        if self.zero_force_input:
            tau_win = np.zeros_like(tau_win)

        # 取网格最后的状态
        left_state = L_qpos_grid[-1].astype(np.float32)
        right_state = R_qpos_grid[-1].astype(np.float32)

        # 图像最邻近匹配
        cam_l_stamps = np.array([t for t, _ in cam_l_list], dtype=np.float64)
        cam_r_stamps = np.array([t for t, _ in cam_r_list], dtype=np.float64)
        idx_l = _nearest_index(cam_l_stamps, t_anchor)
        idx_r = _nearest_index(cam_r_stamps, t_anchor)
        image_l = cam_l_list[idx_l][1].copy()
        image_r = cam_r_list[idx_r][1].copy()

        # 记录对齐漂移
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
        # 返回observation字典
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
    # 推理循环，异步调用policy
    def _infer_step(self):
        if not self.infer_lock.acquire(blocking=False):
            return

        try:
            obs = self._build_observation() #构建观测空间
            if obs is None:
                return

            result = self.client.infer(obs) #调用 WebSocket policy server，获得 actions 序列

            if "actions" not in result:
                self.get_logger().error(f"policy result has no 'actions'. keys={list(result.keys())}")
                return

            actions = np.asarray(result["actions"], dtype=np.float32)

            if actions.ndim == 1:   # 若 actions 是 1D（单帧），则升维到 (1, 14)
                actions = actions[None, :]

            # 维度检查
            if actions.shape[-1] != NUM_JOINTS_TOTAL:
                self.get_logger().error(
                    f"Action dim mismatch: got {actions.shape}, expected last dim {NUM_JOINTS_TOTAL}"
                )
                return
            # 集成到流式缓冲（处理与上一推理结果的平滑过渡）
            self.stream_buffer.integrate_new_chunk(actions, max_k=self.latency_k)
            self.get_logger().info(
                f"Integrated action chunk: shape={actions.shape}, buffer={self.stream_buffer.size()}"
            )

        except Exception as exc:
            self.get_logger().error(f"policy.infer 失败: {exc}")

        finally:
            self.infer_lock.release()

    # 动作平滑与限制：EMA+max_delta+夹爪迟滞
    #对原始 action 做平滑、速度限制、夹爪离散化处理。
    def _smooth_and_limit_action(self, action_14: np.ndarray) -> np.ndarray:
        action_14 = np.asarray(action_14, dtype=np.float32)

        # 第一次进入: 直接采用本帧动作作为参考态.
        # 夹爪如果启用迟滞, 用本帧 raw 判一次初始离散值.
        if self.last_cmd is None:
            init_cmd = action_14.copy()
            if self.gripper_hysteresis_enabled:
                #夹爪离散化
                for i, g in enumerate(GRIPPER_INDICES):
                    raw = float(action_14[g])
                    if raw < self.gripper_thr_low[i]:
                        init_cmd[g] = self.gripper_val_below[i]
                    elif raw > self.gripper_thr_high[i]:
                        init_cmd[g] = self.gripper_val_above[i]
                    else:
                        # 死区里时, 沿用 __init__ 给的初始 last_gripper (val_above, 张开).
                        init_cmd[g] = self.last_gripper[g]
                    self.last_gripper[g] = init_cmd[g]
            self.last_cmd = init_cmd.copy()
            return init_cmd

        out = action_14.copy()

        # 手臂通道: 速度限制 + EMA 
        # 把夹爪索引从手臂掩码中剔除 (仅当迟滞启用时, 否则夹爪也走 EMA).
        arm_mask = np.ones(NUM_JOINTS_TOTAL, dtype=bool)
        if self.gripper_hysteresis_enabled:
            for g in GRIPPER_INDICES:
                arm_mask[g] = False
        # Clip：防止突跳
        delta = np.clip(
            out[arm_mask] - self.last_cmd[arm_mask],
            -self.max_delta_per_step,
            self.max_delta_per_step,
        )
        limited = self.last_cmd[arm_mask] + delta
        # EMA：平滑处理
        # arm_smooth_alpha 控制新值权重
        #   0.0 = 完全保留旧值（最平滑但延迟大）
        #   1.0 = 完全采纳新值（反应快但可能抖动）
        out[arm_mask] = (
            self.arm_smooth_alpha * limited + (1.0 - self.arm_smooth_alpha) * self.last_cmd[arm_mask]
        )

        # 夹爪通道: 迟滞离散化
        if self.gripper_hysteresis_enabled:
            for i, g in enumerate(GRIPPER_INDICES):
                raw = float(action_14[g])
                if raw < self.gripper_thr_low[i]:
                    out[g] = self.gripper_val_below[i]
                elif raw > self.gripper_thr_high[i]:
                    out[g] = self.gripper_val_above[i]
                else:
                    # 死区：保持上一时刻（防抖）
                    out[g] = self.last_gripper[g]
                self.last_gripper[g] = out[g]

        self.last_cmd = out.copy()
        return out

    # 发布命令到机械臂
    # 将平滑后的 14 维 action 分解为左右臂，发布 JointState 消息。
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

    # 控制循环，从缓存中取 action 并发布到机械臂
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

    # 手臂通道单独的 EMA 系数 (不指定则与 --smooth-alpha 相同)
    # 启用夹爪迟滞后, --smooth-alpha 实际只作为 --arm-smooth-alpha 的 fallback, 夹爪不再走 EMA.
    parser.add_argument("--arm-smooth-alpha", type=float, default=None)

    # 夹爪离散化 (迟滞阈值法). 4 个参数任一未设则不启用迟滞, 夹爪走和手臂一样的连续 EMA
    parser.add_argument("--gripper-thr-low", type=str, default=None,
                        help='e.g. "0.025,0.085" 或单值广播 "0.05"')
    parser.add_argument("--gripper-thr-high", type=str, default=None,
                        help='e.g. "0.05,0.09"')
    parser.add_argument("--gripper-val-below", type=str, default=None,
                        help='raw 低于 thr_low 时输出 (一般 = 物理闭合位), e.g. "0.0,0.067"')
    parser.add_argument("--gripper-val-above", type=str, default=None,
                        help='raw 高于 thr_high 时输出 (一般 = 物理张开位), e.g. "0.075,0.095"')

    # 诊断开关: 把送给 policy server 的力窗口整体置零, 用于隔离 ForcePrior 是否依赖
    # 训练时见过、推理时分布漂移的力信号. 不影响夹爪/手臂的其它逻辑.
    parser.add_argument("--zero-force-input", action="store_true",
                        help="把 tau_win 整体置零再送给 policy server (诊断用)")

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
        gripper_thr_low=args.gripper_thr_low,
        gripper_thr_high=args.gripper_thr_high,
        gripper_val_below=args.gripper_val_below,
        gripper_val_above=args.gripper_val_above,
        zero_force_input=args.zero_force_input,
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