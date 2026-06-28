import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

# TYPE_CHECKING 只在静态类型检查阶段生效，运行时不会真正 import Pi0。
# 这样可以避免循环导入：
#   pi0_config.py 需要类型提示 Pi0；
#   pi0.py 又会 import Pi0Config。
if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    """
    Pi0 / Pi0.5 模型配置类。

    这个类只负责描述模型结构、输入输出规格、训练冻结策略等“静态配置”，
    不包含真正的前向传播逻辑。真正的网络结构在 openpi.models.pi0.Pi0 中实现。

    frozen=True 表示配置对象创建后不可修改。
    这样做有两个作用：
      1. 防止训练过程中配置被意外改动；
      2. 便于配置对象作为稳定的、可哈希的结构使用。
    """
    # 模型计算使用的数据类型，默认为 bfloat16 以节省显存并加速训练/推理
    dtype: str = "bfloat16"
    # PaliGemma-side Gemma expert 的变体。
    # 它主要处理 prefix 侧上下文：
    #   - 视觉 token
    #   - 语言 token
    #   - pi0.5 中可能还包括离散化后的 state token
    #
    # 这里的 "gemma_2b" 表示使用 2B 规模的 Gemma 配置。
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # Action Expert 的 Gemma 变体。
    # 它主要处理 suffix 侧动作 token。
    #
    # 在 pi0 / pi0.5 结构中，模型通常是 dual-expert：
    #   - 一个 expert 处理 prefix context；
    #   - 一个 expert 处理 action suffix。
    #
    # 这里默认使用较小的 300M action expert。
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # 设置模型特定的默认值。
    
    # 动作空间的维度，即每个时间步输出的动作向量长度
    action_dim: int = 32
    
    # 动作 horizon，即模型一次性预测未来的动作步数
    action_horizon: int = 50
    
    # 最大令牌长度，用于限制输入 prompt 的长度。
    # 如果未指定，将在 __post_init__ 中根据 pi05 标志自动设置
    max_token_len: int = None  # type: ignore
    
    # 是否启用 Pi05 模式。
    # Pi05 与 Pi0 的主要区别：
    # 1. 状态输入 (state input) 被处理为离散的语言令牌 (discrete language tokens)，
    #    而不是作为后缀部分的连续向量输入。
    # 2. Action Expert 使用 adaRMSNorm 技术来注入流匹配 (flow matching) 的时间步信息，
    #    以更好地建模动作分布。
    pi05: bool = False
    # 此配置选项不直接被模型使用，但会被 ModelTransformFactory 读取。
    # 如果为 True：
    #   robot state 会被 transform 转成离散 token，并并入 prompt/prefix。
    #
    # 如果为 False：
    #   robot state 更可能作为连续 state vector 输入模型。
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        """
        dataclass 初始化后自动调用。
        因为 frozen=True，不能直接写：
            self.max_token_len = ...
        所以这里使用 object.__setattr__ 绕过 frozen 限制，
        只在初始化阶段补全默认值。
        """
        # 如果没有手动指定 max_token_len，则根据 pi0 / pi0.5 自动设置。
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        # 如果没有手动指定 discrete_state_input，则默认与 pi05 保持一致。
        # 也就是说：
        #   pi0.5 默认使用离散状态输入；
        #   pi0 默认不使用离散状态输入。
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """
        定义模型期望的输入输出 shape / dtype。
        这个函数不创建真实数据，只创建 ShapeDtypeStruct。
        它主要用于：
          - JAX jit / shape tracing；
          - 初始化 fake observation；
          - 检查数据 pipeline 输出是否与模型匹配。

        返回：
          observation_spec:
            模型观测输入规格。
          action_spec:
            训练时 ground-truth action 的规格。
        """
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """返回基于模型配置的冻结过滤器。"""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # 如果只冻结 Gemma 参数，则排除动作专家参数。
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # 如果使用了任何 LoRA，则排除所有 LoRA 参数。
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
