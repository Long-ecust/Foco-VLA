#!/usr/bin/env python3
"""

1. **时间戳对齐**: HDF5 里 joint_feedback_filtered 比 image 慢约 100ms, end_pose 慢
   约 300ms. 这里以 image[:, 0] (前置相机) 时间戳为对齐基准, 把所有连续信号 (qpos,
   qvel, filtered_effort) **线性插值** 到 target_fps 的规则网格上, 图像按最近邻取帧.

2. **采样率统一**: 录制的实际 fps ≈ 27 而非名义 30, 重采到 target_fps (默认 30) 后
   网格 dt 完全规则, lerobot 的 delta_timestamps 不需要 tolerance fallback.

3. **action = 下一帧 qpos**: HDF5 没记录 teleop 指令流; 用 future qpos 作 action target
   (位置策略的标准做法). 模型训练时 lerobot 通过 delta_timestamps 自动堆叠成
   (action_horizon, 14) 的 chunk.

4. **Per-joint sign / scale 校正 (hook only)**: 当前用 identity. 未来:
   - j5 符号翻转: 等标定数据确认后填 ±1 向量
   - 3x effort scale: 等悬挂质量标定后填每关节 α_j
   分两个 npz 文件加载, 不动 HDF5、不重新跑 conversion.

5. **不预计算 Δτ / τ_comp**:
   - Δτ 由 PerJointForceTokenizer 从 W 帧历史自学, 存它就是冗余 + 把噪声固化
   - τ_comp 等 URDF 标定参数齐了后在 dataloader / inspector 层算, 不预 bake
     (URDF 参数会迭代多次, 任何预 bake 都会需要重转 200G 数据)

Usage:
    uv run python examples/piper_bimanual/conversion/convert_longvla_hdf5_to_lerobot.py \\
        --raw-dir examples/datasets/handover0507--3 \\
        --repo-id longvla_handover_v1 \\
        [--target-fps 30] [--mode image] [--max-episodes 1]
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import logging
import pathlib
import shutil

import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# 修改这里前请同时检查:
#   - src/openpi/models/longvla.py        : LongVLAConfig
#   - src/openpi/policies/longvla_policy.py : LongVLAInputs
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
NUM_ARMS = 2
PER_ARM_JOINTS = 7                                  # 6 关节 + gripper
NUM_JOINTS_TOTAL = NUM_ARMS * PER_ARM_JOINTS        # 14
LEFT_ARM_SLICE = slice(0, PER_ARM_JOINTS)
RIGHT_ARM_SLICE = slice(PER_ARM_JOINTS, NUM_JOINTS_TOTAL)
# robot_state 每只手 7 维 = qpos (6 关节 + gripper 位置). 与 per_arm_force_dim 完全对齐,
# state 通道 i 和 force.qpos 通道 i 含义一致.
PER_ARM_STATE_DIM = 7

# 14 维 qpos 布局: [L_j1..L_j6, L_gripper, R_j1..R_j6, R_gripper], gripper 在 (6, 13).
# 夹爪 raw 分布是双峰 (例如 L: 0/0.075, R: 0.067/0.095), 对它做线性插值会在两个稳态之间
# 凭空插出 0.03/0.04/... 这种训练 raw 数据里**不存在**的"幻影值", 模型会被强制学这些幻影
# 值, 进而在推理时输出双峰之间的中间夹爪指令, 表现为夹爪抖动 / 抓不稳. 改用 step-hold
# (保持上一个观测值) 避免创造新值, 关节通道仍走 linear (关节物理上确实在两次采样间平滑
# 移动, linear interp 是合理近似). qvel/tau 的夹爪通道同样 step-hold——夹爪稳态时
# qvel/tau 都是 0, transition 帧极少, linear interp 会把瞬时尖峰拉成连续过渡, 失真.
GRIPPER_CHANNELS: tuple[int, ...] = (PER_ARM_JOINTS - 1, NUM_JOINTS_TOTAL - 1)  # (6, 13)


@dataclasses.dataclass(frozen=True)
class JointCorrection:
    """每关节符号 / 标度校正系数, 全 1 时是 no-op.

    这些校正本质上应当来自外部标定实验:
      - qpos_sign / qvel_sign : URDF 与 SDK qpos 约定的轴向差异 (例如 j5 符号翻转)
      - effort_scale          : 厂家 Kt 未标定造成的标度偏差 (实测 ~1/3.18x)
    保留作为 hook 的目的是: 标定一次, 改一个 npz, 后续所有 conversion 自动应用,
    不需要重录数据.
    """
    qpos_sign: np.ndarray
    qvel_sign: np.ndarray
    effort_scale: np.ndarray

    @classmethod
    def identity(cls) -> "JointCorrection":
        return cls(
            qpos_sign=np.ones(NUM_JOINTS_TOTAL, dtype=np.float32),
            qvel_sign=np.ones(NUM_JOINTS_TOTAL, dtype=np.float32),
            effort_scale=np.ones(NUM_JOINTS_TOTAL, dtype=np.float32),
        )

    @classmethod
    def from_npz(cls, path: pathlib.Path) -> "JointCorrection":
        """从 npz 文件加载 (key 名: qpos_sign / qvel_sign / effort_scale)."""
        data = np.load(path)
        c = cls(
            qpos_sign=data["qpos_sign"].astype(np.float32),
            qvel_sign=data["qvel_sign"].astype(np.float32),
            effort_scale=data["effort_scale"].astype(np.float32),
        )
        for name, arr in (("qpos_sign", c.qpos_sign), ("qvel_sign", c.qvel_sign), ("effort_scale", c.effort_scale)):
            if arr.shape != (NUM_JOINTS_TOTAL,):
                raise ValueError(f"{name} shape {arr.shape}, expected ({NUM_JOINTS_TOTAL},)")
        return c


# ============================================================================
# 时间戳与重采样
# ============================================================================

def _build_grid_intersection(sources: list[np.ndarray], target_fps: float) -> np.ndarray:
    """在多个源时间戳序列的**交集**范围内构造规则采样网格.

    Piper 录制管线里, joint feedback 话题通常比前置相机晚 ~300-500ms 才上线,
    若只按 cam_f 范围建 grid, 早期 ~14 帧的 grid 点落在 joint_ts[0] 之前,
    np.interp 会静默外推为 joint 第一帧的值, 写入大段假数据.
    用所有源时间戳的交集建 grid, 自然裁掉这段未对齐前缀.
    """
    t_start = max(float(s[0]) for s in sources)
    t_end = min(float(s[-1]) for s in sources)
    if t_end <= t_start:
        return np.empty(0, dtype=np.float64)
    duration = t_end - t_start
    n_frames = int(np.floor(duration * target_fps)) + 1
    return t_start + np.arange(n_frames) / target_fps


def _interp_to_grid(
    grid_t: np.ndarray,
    src_t: np.ndarray,
    src_x: np.ndarray,
    *,
    stephold_channels: tuple[int, ...] = (),
) -> np.ndarray:
    """把 src_x (在 src_t 上采样的多通道连续信号) 重采样到 grid_t.

    src_t : (T_src,) 单调递增
    src_x : (T_src, D)
    grid_t: (T_grid,)
    stephold_channels: 这些通道用 step-hold (保持上一个观测), 其余用 linear interp.
                       双峰离散类信号 (如夹爪开/闭) 必须 step-hold, 否则 linear interp
                       会创造原始数据里不存在的中间值.
    返回    (T_grid, D)

    注意: np.interp 在超出 src_t 范围的 grid_t 上**静默外推到边界值**, 不会报错.
    这是已知的潜在静默 bug, 调用方在数据进来之前应该做 coverage 检查 (见 convert_episode).
    """
    out = np.empty((len(grid_t), src_x.shape[1]), dtype=np.float32)
    stephold_set = set(stephold_channels)
    if stephold_set:
        # 对每个 grid 点找最近的"过去"原始样本, side='right' 保证 grid 点正好落在
        # 样本上时取该样本本身. 越界用 clip 兜底.
        sh_idx = np.searchsorted(src_t, grid_t, side="right") - 1
        sh_idx = np.clip(sh_idx, 0, len(src_t) - 1)
    for d in range(src_x.shape[1]):
        if d in stephold_set:
            out[:, d] = src_x[sh_idx, d]
        else:
            out[:, d] = np.interp(grid_t, src_t, src_x[:, d])
    return out


def _coverage_ok(grid_t: np.ndarray, src_t: np.ndarray, *, tol: float = 0.0) -> bool:
    """grid_t 是否完全被 src_t 覆盖 (允许 tol 秒外推容差)."""
    return src_t[0] <= grid_t[0] + tol and grid_t[-1] - tol <= src_t[-1]


def _nearest_indices(grid_t: np.ndarray, image_ts: np.ndarray) -> np.ndarray:
    """对 grid 中每个时间, 找最近的图像帧 index. 图像是离散帧, 不能插值."""
    # 用 searchsorted + 邻接比较, O((T_grid + T_image) log T_image)
    idx = np.searchsorted(image_ts, grid_t)
    idx = np.clip(idx, 1, len(image_ts) - 1)
    left = image_ts[idx - 1]
    right = image_ts[idx]
    pick_left = (grid_t - left) < (right - grid_t)
    return np.where(pick_left, idx - 1, idx).astype(np.int64)


# ============================================================================
# 数据集创建 / 单 episode 转换
# ============================================================================

JOINT_COLUMN_NAMES = [f"L_{n}" for n in ARM_JOINT_NAMES] + [f"R_{n}" for n in ARM_JOINT_NAMES]
LEFT_STATE_NAMES = [f"L_{n}" for n in ARM_JOINT_NAMES]   # 7 维 = 6 关节 + gripper 位置
RIGHT_STATE_NAMES = [f"R_{n}" for n in ARM_JOINT_NAMES]


def create_empty_dataset(
    repo_id: str, target_fps: int, mode: str, image_writer_processes: int, image_writer_threads: int
) -> LeRobotDataset:
    features = {
        # ---- 当前帧 robot_state (每只手 7 维 = 6 关节 qpos + gripper 位置) ----
        "observation.left_state": {
            "dtype": "float32", "shape": (PER_ARM_STATE_DIM,), "names": [LEFT_STATE_NAMES],
        },
        "observation.right_state": {
            "dtype": "float32", "shape": (PER_ARM_STATE_DIM,), "names": [RIGHT_STATE_NAMES],
        },
        # ---- Force history 三通道 (单帧, 采样时通过 delta_timestamps 取 W 帧) ----
        "observation.qpos": {
            "dtype": "float32", "shape": (NUM_JOINTS_TOTAL,), "names": [JOINT_COLUMN_NAMES],
        },
        "observation.qvel": {
            "dtype": "float32", "shape": (NUM_JOINTS_TOTAL,), "names": [JOINT_COLUMN_NAMES],
        },
        "observation.filtered_effort": {
            "dtype": "float32", "shape": (NUM_JOINTS_TOTAL,), "names": [JOINT_COLUMN_NAMES],
        },
        # ---- 三个场景相机 (键名不带 wrist, 保留 RandomCrop+Rotate 增广) ----
        "observation.image": {
            "dtype": mode, "shape": (3, 480, 640), "names": [["channels", "height", "width"]],
        },
        "observation.left_image": {
            "dtype": mode, "shape": (3, 480, 640), "names": [["channels", "height", "width"]],
        },
        "observation.right_image": {
            "dtype": mode, "shape": (3, 480, 640), "names": [["channels", "height", "width"]],
        },
        # ---- action target = 下一帧 qpos ----
        "action": {
            "dtype": "float32", "shape": (NUM_JOINTS_TOTAL,), "names": [JOINT_COLUMN_NAMES],
        },
        # 注: timestamp 不需要在 features 里声明, lerobot 会按 frame_index / fps 自动填充.
        # task 字段则在 add_frame 时按帧传入, 不需要进 features.
    }

    if (HF_LEROBOT_HOME / repo_id).exists():
        log.warning(f"Removing existing dataset at {HF_LEROBOT_HOME / repo_id}")
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(target_fps),
        robot_type="piper_bimanual_right_only",
        features=features,
        use_videos=(mode == "video"),
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def convert_episode(
    dataset: LeRobotDataset,
    ep_path: pathlib.Path,
    target_fps: float,
    correction: JointCorrection,
    default_task: str | None,
) -> int:
    """转换一个 HDF5 episode 到 lerobot. 返回写入的帧数."""
    with h5py.File(ep_path, "r") as ep:
        # 数值字段 (相对小, 一次性加载)
        qpos_raw = np.asarray(ep["observations/qpos"], dtype=np.float32)
        qvel_raw = np.asarray(ep["observations/qvel"], dtype=np.float32)
        tau_raw = np.asarray(ep["observations/filtered_effort"], dtype=np.float32)

        # 时间戳
        ts_image = np.asarray(ep["timestamps/image"], dtype=np.float64)               # (T, 3)
        ts_joint_filt = np.asarray(ep["timestamps/joint_feedback_filtered"], dtype=np.float64)  # (T, 2)

        # 任务文本
        task = default_task or str(ep.attrs.get("prompt", ep.attrs.get("task", "bimanual handover")))

        # 三相机各自时间戳列 (与 meta/camera_names 顺序对应)
        cam_names = [n.decode() for n in ep["meta/camera_names"][...]]
        # 预期 ['camera_f', 'camera_l', 'camera_r']
        ts_cam_f = ts_image[:, cam_names.index("camera_f")]
        ts_cam_l = ts_image[:, cam_names.index("camera_l")]
        ts_cam_r = ts_image[:, cam_names.index("camera_r")]

        # 关节信号双臂各有一个时间戳 (left, right); 这里近似用均值作为统一时间.
        # 经检查左右臂时间戳差异通常 < 5ms, 远小于 1 帧, 取均值不影响插值精度.
        joint_ts = ts_joint_filt.mean(axis=1)

        # ---- 构建规则采样网格 (取所有源时间戳的交集, 避免边界外推) ----
        # 录制管线里 joint feedback 比 cam_f 晚 ~300-500ms 上线, 单用 cam_f 范围建 grid
        # 会让早期 ~14 帧落在 joint_ts[0] 之前, 触发 np.interp 静默外推. 用交集自然裁掉.
        grid_t = _build_grid_intersection(
            [ts_cam_f, ts_cam_l, ts_cam_r, joint_ts], target_fps
        )
        if len(grid_t) < 2:
            log.warning(f"Episode {ep_path.name} too short ({len(grid_t)} frames), skipping")
            return 0

        # ---- coverage 检查 (交集 grid 下理论上不应触发, 触发说明数据异常) ----
        if not _coverage_ok(grid_t, joint_ts):
            log.warning(
                f"Episode {ep_path.name}: joint_ts 未完全覆盖 grid_t "
                f"(joint [{joint_ts[0]:.3f}, {joint_ts[-1]:.3f}] vs "
                f"grid [{grid_t[0]:.3f}, {grid_t[-1]:.3f}]), 将出现边界值外推"
            )
        for cam_name, cam_ts in (("camera_l", ts_cam_l), ("camera_r", ts_cam_r)):
            if not _coverage_ok(grid_t, cam_ts):
                log.warning(
                    f"Episode {ep_path.name}: {cam_name} 未完全覆盖 grid_t "
                    f"({cam_name} [{cam_ts[0]:.3f}, {cam_ts[-1]:.3f}] vs "
                    f"grid [{grid_t[0]:.3f}, {grid_t[-1]:.3f}]), 将出现重复边界帧"
                )

        # ---- 重采样连续信号到网格 ----
        # 关节通道 linear (物理上平滑), 夹爪通道 step-hold (双峰分布, 不能造中间值).
        qpos_grid = _interp_to_grid(grid_t, joint_ts, qpos_raw, stephold_channels=GRIPPER_CHANNELS)
        qvel_grid = _interp_to_grid(grid_t, joint_ts, qvel_raw, stephold_channels=GRIPPER_CHANNELS)
        tau_grid = _interp_to_grid(grid_t, joint_ts, tau_raw, stephold_channels=GRIPPER_CHANNELS)

        # ---- 应用符号 / 标度校正 ----
        # 校正只在数值上做, 不动数据 dtype 也不重新计算什么; 校正为 identity 时是 no-op.
        qpos_grid = qpos_grid * correction.qpos_sign
        qvel_grid = qvel_grid * correction.qvel_sign
        tau_grid = tau_grid * correction.effort_scale

        # ---- 构建 robot_state (每只手 7 维 = qpos: 6 关节 + gripper 位置) ----
        # 仅取 qpos, 不再拼 gripper 速度——保持与 LongVLAConfig.per_arm_state_dim=7 一致,
        # 让 state[i] 和 force_history[i, t, 0] (qpos 通道) 在 i 维度上语义对齐.
        left_state = qpos_grid[:, LEFT_ARM_SLICE].astype(np.float32)    # (T, 7)
        right_state = qpos_grid[:, RIGHT_ARM_SLICE].astype(np.float32)

        # ---- action target = 下一帧 qpos ----
        # 训练时 lerobot 通过 action 的 delta_timestamps [0, 1/fps, ..., (H-1)/fps] 取 action_horizon
        # 帧, 即得到 [qpos[t+1], qpos[t+2], ..., qpos[t+H]]. 最后一帧没有 next qpos, 丢弃.
        action = qpos_grid[1:]   # (T-1, 14)
        n_frames = action.shape[0]

        # ---- 图像近邻索引 ----
        idx_cam_f = _nearest_indices(grid_t[:n_frames], ts_cam_f)
        idx_cam_l = _nearest_indices(grid_t[:n_frames], ts_cam_l)
        idx_cam_r = _nearest_indices(grid_t[:n_frames], ts_cam_r)

        cam_f_ds = ep["observations/images/camera_f"]
        cam_l_ds = ep["observations/images/camera_l"]
        cam_r_ds = ep["observations/images/camera_r"]

        for t in range(n_frames):
            frame = {
                "observation.left_state": left_state[t],
                "observation.right_state": right_state[t],
                "observation.qpos": qpos_grid[t].astype(np.float32),
                "observation.qvel": qvel_grid[t].astype(np.float32),
                "observation.filtered_effort": tau_grid[t].astype(np.float32),
                "observation.image": np.asarray(cam_f_ds[idx_cam_f[t]]),
                "observation.left_image": np.asarray(cam_l_ds[idx_cam_l[t]]),
                "observation.right_image": np.asarray(cam_r_ds[idx_cam_r[t]]),
                "action": action[t].astype(np.float32),
                # task 按帧传入, lerobot 会做 string -> task_index 的映射
                "task": task,
            }
            dataset.add_frame(frame)
        dataset.save_episode()
        return n_frames


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert longvla_raw_v2 HDF5 episodes to lerobot for LongVLA training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--raw-dir", required=True, help="Directory containing episode_*.hdf5 files")
    ap.add_argument("--repo-id", required=True, help="Output lerobot dataset repo id")
    ap.add_argument("--target-fps", type=float, default=30.0, help="Target resampling fps (default 30)")
    ap.add_argument("--task", default=None, help="Override task prompt (else read from HDF5 attrs)")
    ap.add_argument("--mode", choices=["video", "image"], default="image")
    ap.add_argument("--max-episodes", type=int, default=None, help="Limit episodes (for debug)")
    ap.add_argument(
        "--correction-npz",
        type=pathlib.Path,
        default=None,
        help="Path to a npz with keys (qpos_sign, qvel_sign, effort_scale). "
        "Defaults to identity (no correction).",
    )
    ap.add_argument("--image-writer-processes", type=int, default=8)
    ap.add_argument("--image-writer-threads", type=int, default=4)
    args = ap.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    # 只匹配真正的 hdf5, 跳过 .larkcache 等下载中文件
    files = sorted(p for p in raw_dir.glob("episode_*.hdf5") if not p.name.endswith(".larkcache"))
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 found under {raw_dir}")
    log.info(f"Found {len(files)} episode(s) in {raw_dir}")

    if args.correction_npz is not None:
        correction = JointCorrection.from_npz(args.correction_npz)
        log.info(f"Loaded joint correction from {args.correction_npz}")
        log.info(f"  qpos_sign    = {correction.qpos_sign}")
        log.info(f"  qvel_sign    = {correction.qvel_sign}")
        log.info(f"  effort_scale = {correction.effort_scale}")
    else:
        correction = JointCorrection.identity()
        log.info("Using identity joint correction (no sign / scale fix). "
                 "Pass --correction-npz once calibration data is available.")

    dataset = create_empty_dataset(
        args.repo_id,
        target_fps=int(round(args.target_fps)),
        mode=args.mode,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    for i, ep_path in enumerate(tqdm.tqdm(files, desc="Episodes"), start=1):
        try:
            n = convert_episode(dataset, ep_path, args.target_fps, correction, args.task)
            total_frames += n
            log.info(f"Episode {i}/{len(files)} ({ep_path.name}): {n} frames, total={total_frames}")
        except Exception as e:
            log.error(f"Failed on {ep_path.name}: {e}", exc_info=True)
            continue
        finally:
            # 强制释放资源,避免内存/文件句柄累积
            gc.collect()

    log.info(f"Conversion done. Total frames written: {total_frames}")


if __name__ == "__main__":
    main()
