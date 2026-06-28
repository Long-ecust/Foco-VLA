from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# 与 LongVLAConfig 默认值保持一致；如果将来 schema 改动这里也要跟着改
DEFAULT_PER_ARM_STATE_DIM = 7
DEFAULT_NUM_JOINTS = 14   # 双臂 (6 关节 + gripper) × 2
DEFAULT_FORCE_WINDOW = 16
DEFAULT_FORCE_CHANNELS = 3  # (qpos, qvel, τ_filtered)


def make_longvla_example() -> dict:
    #fake 样本，供 norm-stats 收集 / 本地 dry-run 使用.
    W = DEFAULT_FORCE_WINDOW
    K = DEFAULT_NUM_JOINTS
    return {
        "observation/qpos": np.random.randn(W, K).astype(np.float32),
        "observation/qvel": np.random.randn(W, K).astype(np.float32),
        "observation/filtered_effort": np.random.randn(W, K).astype(np.float32),
        # 当前数据右臂单独工作, 左臂全 0; schema 双臂保持向前兼容.
        "observation/left_state": np.zeros(DEFAULT_PER_ARM_STATE_DIM, dtype=np.float32),
        "observation/right_state": np.random.randn(DEFAULT_PER_ARM_STATE_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "hand the bottle of tea to me",
    }


def _parse_image(image) -> np.ndarray:
    """统一图像格式: 转为 HWC uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:  # CHW → HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LongVLAInputs(transforms.DataTransformFn):

    model_type: _model.ModelType
    per_arm_state_dim: int = DEFAULT_PER_ARM_STATE_DIM
    num_joints: int = DEFAULT_NUM_JOINTS
    force_window: int = DEFAULT_FORCE_WINDOW
    force_channels: int = DEFAULT_FORCE_CHANNELS

    def __call__(self, data: dict) -> dict:
        # ---- 图像 ----
        base_image = _parse_image(data["observation/image"])
        left_image = _parse_image(data["observation/left_image"])
        right_image = _parse_image(data["observation/right_image"])

        # ---- 当前帧 robot_state ----
        # 缺失的 arm 用零填充 (例如当前数据只有右臂工作时)
        zero_arm_state = np.zeros(self.per_arm_state_dim, dtype=np.float32)
        left_state = np.asarray(
            data.get("observation/left_state", zero_arm_state), dtype=np.float32
        )
        right_state = np.asarray(
            data.get("observation/right_state", zero_arm_state), dtype=np.float32
        )
        if left_state.shape != (self.per_arm_state_dim,):
            raise ValueError(
                f"left_state shape {left_state.shape}, expected ({self.per_arm_state_dim},)"
            )
        if right_state.shape != (self.per_arm_state_dim,):
            raise ValueError(
                f"right_state shape {right_state.shape}, expected ({self.per_arm_state_dim},)"
            )
        robot_state = np.concatenate([left_state, right_state], axis=-1) 

        # ---- Force history (W frames of qpos / qvel / τ) ----
        # 三个通道各为 (W, K) 形状，stack 成 (W, K, 3)，转置成 (K, W, 3)，flatten 后
        # 模型端 reshape(B, K, W, C) 能恢复原结构。通道顺序: (qpos, qvel, τ_filtered).
        qpos = np.asarray(data["observation/qpos"], dtype=np.float32)
        qvel = np.asarray(data["observation/qvel"], dtype=np.float32)
        tau = np.asarray(data["observation/filtered_effort"], dtype=np.float32)
        expected_shape = (self.force_window, self.num_joints)
        for name, arr in (("qpos", qpos), ("qvel", qvel), ("filtered_effort", tau)):
            if arr.shape != expected_shape:
                raise ValueError(
                    f"observation/{name} shape {arr.shape}, expected {expected_shape}. "
                    "Make sure lerobot delta_timestamps is configured to return "
                    f"force_window={self.force_window} historical frames."
                )
        # w表示关注历史多少帧时数据，C表示每个真时数据的维度，K表示关节数，w=16，C=3，K=14
        # 3个通道分别是 qpos, qvel, tau_filtered
        force_history_wkc = np.stack([qpos, qvel, tau], axis=-1)            # (W, K, 3)
        force_history_kwc = np.transpose(force_history_wkc, (1, 0, 2))      # (K, W, 3)

        # state 只装 robot_state (~16 维, 会被 discrete tokenization 处理);
        # force_history 走独立字段, 不进 prompt 离散化.
        inputs = {
            "state": robot_state.astype(np.float32),                 # (16,)
            "force_history": force_history_kwc.astype(np.float32),   # (K, W, C) = (14, 16, 3)
            "image": {
                "base_0_rgb": base_image,
                "left_0_rgb": left_image,
                "right_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_0_rgb": np.True_,
                "right_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class LongVLAOutputs(transforms.DataTransformFn):
    #LongVLA输出维度是32维，因为32维是Transformer/Gemma喜欢固定维度
    #把模型输出 (padded 到 action_dim=32) 切回真实动作维度 (默认 14).
    action_dims: int = 14
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dims])}
