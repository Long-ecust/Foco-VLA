from __future__ import annotations

import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import posemb_sincos
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


# LongVLA 的相机 key 命名。
# 注意：与 pi0.5 默认的 ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb") 不同——
# openpi 的 preprocess_observation 通过 "wrist" 子串决定是否做 RandomCrop+Rotate 增广
# (默认 wrist 相机视角敏感不做这两个增广)。我们的 left/right 是场景三视角相机，必须
# 不带 "wrist" 才能获得完整增广序列。模型在 compute_loss / sample_actions 中通过显式
# image_keys 参数把这些 key 透传给 preprocess_observation。
LONGVLA_IMAGE_KEYS = ("base_0_rgb", "left_0_rgb", "right_0_rgb")


@dataclasses.dataclass(frozen=True)
class LongVLAConfig(_model.BaseModelConfig):
    # LongVLA 训练/推理配置

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # action chunk 维度。双臂 (6 关节 + gripper) × 2 = 14；模型对齐到 32 做 padding，
    # 真实使用时通过 LongVLAOutputs 取前 14 维。
    action_dim: int = 32
    action_horizon: int = 25    #pi0.5是50
    max_token_len: int = 200    #pi0.5是200

    # 沿用 pi0.5 的双专家 Gemma 主干和 flow matching 训练范式。
    pi05: bool = True
    discrete_state_input: bool = True

    # 双臂结构。即使当前数据右臂单独工作，左臂占位为零，schema 保持双臂便于未来扩展。
    num_arms: int = 2
    # 每只手的 robot_state 维度: 6 关节位置 + 1 gripper 位置 = 7。与 per_arm_force_dim
    # 完全对齐，state 通道 i 和 force.qpos 通道 i 含义一致 (i ∈ [0,6])。
    per_arm_state_dim: int = 7
    # 每只手的力通道数（即 force token 数）；Piper 7 自由度（含 gripper）。
    per_arm_force_dim: int = 7

    # ---- 力分支结构 ----
    # 每个关节 token 看的历史窗口长度（帧）。30 fps 下 W=16 ≈ 0.53 s，
    # 足以覆盖一次接触事件的发生过程。
    force_window: int = 16
    # 每个关节每帧的通道数。布局: (qpos, qvel, τ_filtered) → 3。
    force_channels: int = 3
    # ForcePrior 内部 MLP 隐藏维度。
    force_prior_hidden: int = 256
    # PerJointForceTokenizer 内部 MLP 隐藏维度。
    joint_token_hidden: int = 128

    # 是否给 ForcePrior 喂视觉特征。
    # False (默认): prior 只看 (lang_pool, qpos_t) —— 信息瓶颈版本，抗 prior collapse
    # True:        prior 看 (vision_pool, lang_pool, qpos_t) —— 仅用于 paper ablation 对照
    prior_uses_vision: bool = False

    # 是否给 PerJointForceTokenizer 的输出加 (arm_id, joint_id) 嵌入.
    # 共享 MLP 本身对 K 个关节是各向同性的, 没有这个 ID 信息时, 关节区分只能靠
    # prefix 内 position embedding 隐式给出 (而且 PE 是绝对位置, 不携带"我是 gripper"
    # 这种语义). 加上 ID 嵌入后, 模型能立刻区分 (左/右臂) × (joint1..joint7), 利于
    # 小数据下 force branch 更快学到关节特异的力分布. False = ablation 对照.
    use_joint_id_embedding: bool = True

    # 辅助损失：让 ForcePrior 通过对实测 τ 的 MSE 学出任务条件化的力先验。
    aux_anomaly_weight: float = 0.01

    # === Ablation flags (论文 Table 1 每行一个 flag 组合) ===
    #
    # A2: 整条 ForcePrior 分支砍掉. True 时:
    #   - 不构造 force_prior 模块 (省参数 + forward 开销)
    #   - tau_hat 强制为零向量 → tokenizer 的 prior-derived extras 退化为 (0, |tau_t|)
    #   - aux_anomaly_loss 强制为 0 (不进 total_loss, 但保留在 aux metrics 供观察)
    # 与 aux_anomaly_weight=0 训练的区别: 后者 prior 模块还在跑 forward, 只是没梯度
    # 信号 (fc_out 零初始化 → 永远输出 0). disable_force_prior 是从结构上彻底拿掉, 验证
    # "task-conditioned prior 这个机制是否真的必要"——比 aux_weight 切到 0 更干净.
    disable_force_prior: bool = False

    # A5: force token 的粒度.
    #   "perjoint" (默认): 每个关节产出一个 token → K 个 prefix token. 这是 LongVLA
    #                      相对 TA-VLA 的核心新颖性轴之一.
    #   "single":          把 K 个关节的 (历史 + prior extras) 全部拼成一个向量过 MLP,
    #                      只产出 1 个 prefix token. 仍在 encoder 侧、仍带 task-conditioned
    #                      prior——唯一变量是"逐关节 vs 单 token". 用来直接隔离 per-joint
    #                      tokenization 本身的贡献 (对标 TA-VLA EXPERT_HIS_C 的单 token
    #                      聚合, 但搬到 encoder 侧并加 prior). single 模式下 joint/arm ID
    #                      embedding 无意义, 自动忽略.
    force_token_mode: str = "perjoint"

    @property
    def num_joints(self) -> int:
        # 所有 force token 的总数 K，例如 num_arms=2, per_arm_force_dim=7 → K=14.
        return self.num_arms * self.per_arm_force_dim

    @property
    def robot_state_dim(self) -> int:
        # robot_state 部分在 obs.state 中占据的维数.
        return self.num_arms * self.per_arm_state_dim

    @property
    def force_history_size(self) -> int:
        # force history 总元素数 K * W * C (用于校验 / 调试).
        return self.num_joints * self.force_window * self.force_channels
    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "LongVLA":
        return LongVLA(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={key: image_spec for key in LONGVLA_IMAGE_KEYS},
                image_masks={key: image_mask_spec for key in LONGVLA_IMAGE_KEYS},
                # state 经过 PadStatesAndActions 会被 pad 到 action_dim, 这里直接声明 padded size.
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                force_history=jax.ShapeDtypeStruct(
                    [batch_size, self.num_joints, self.force_window, self.force_channels],
                    jnp.float32,
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        # LoRA 训练时使用：决定哪些参数冻结、哪些只训练 LoRA 增量.
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True

        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class ForcePrior(nnx.Module):
    # 任务条件化的关节力矩先验.
    # 它是一个 NNX 模块，本质是一个小 MLP。
    def __init__(
        self,
        lang_dim: int,  # 语言token embedding维度，是PaliGemma hidden width
        num_joints: int,    # 关节数，默认是14
        hidden: int,    # 隐藏层维度，默认是256
        *,
        vision_dim: int | None = None,  # 是否加入视觉池化特征。None 表示不使用视觉。
        rngs: nnx.Rngs,
    ):
        self.use_vision = vision_dim is not None
        in_dim = (vision_dim if self.use_vision else 0) + lang_dim + num_joints #如果不用视觉的话，in_dim = lang_dim + 14
        self.fc1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        # fc_out 用零初始化: 训练 step 0 时 τ̂ ≡ 0, 等价于"无力先验" baseline.
        # 这样 (a) aux MSE loss 从 τ 的方向开始把 prior 推到真实期望; (b) mismatch
        # 初始等于 |τ|, 模型先用 raw force 工作, 随着 prior 学到任务期望逐渐 carve out 异常.
        # 等价于 LoRA B 矩阵零初始化的思路 — 新分支在 pretrained 骨干上"先无害再起作用".
        self.fc_out = nnx.Linear(
            hidden, num_joints,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(
        self,
        lang_pool: jax.Array,
        qpos_t: jax.Array,
        vision_pool: jax.Array | None = None,
    ) -> jax.Array:
        if self.use_vision:
            assert vision_pool is not None, "ForcePrior 构造时启用了视觉，调用必须传 vision_pool"
            x = jnp.concatenate([vision_pool, lang_pool, qpos_t], axis=-1)
        else:
            x = jnp.concatenate([lang_pool, qpos_t], axis=-1)
        x = nnx.swish(self.fc1(x))
        x = nnx.swish(self.fc2(x))
        return self.fc_out(x)

    #把每个关节的历史轨迹编码成一个 token，进入 prefix.
    #让模型从所有数据中学一个通用的"关节力时序编码器"。
class PerJointForceTokenizer(nnx.Module):
    def __init__(
        self,
        num_joints: int,
        window: int,
        channels: int,
        embed_dim: int,
        hidden: int,
        *,
        num_arms: int = 1,
        use_id_embedding: bool = True,
        token_mode: str = "perjoint",
        rngs: nnx.Rngs,
    ):
        self.num_joints = num_joints
        self.window = window
        self.channels = channels
        self.num_arms = num_arms
        if num_joints % num_arms != 0:
            raise ValueError(f"num_joints={num_joints} must be divisible by num_arms={num_arms}")
        self.per_arm_joints = num_joints // num_arms
        if token_mode not in ("perjoint", "single"):
            raise ValueError(f"token_mode must be 'perjoint' or 'single', got {token_mode!r}")
        self.token_mode = token_mode
        # single 模式没有逐关节 token, ID embedding 无处可加; 强制关闭以免误用.
        self.use_id_embedding = use_id_embedding and token_mode == "perjoint"
        # 输入维度:
        #   perjoint: 单个关节的 W×C 帧扁平化 + 2 维当前帧标量 (τ̂_j, |τ_t-τ̂_j|), MLP 在 K 上共享.
        #   single:   K 个关节全部拼接 = K×(W×C) 历史 + 2K 维 prior extras, 一次性出 1 个 token.
        if token_mode == "perjoint":
            in_dim = window * channels + 2
        else:
            in_dim = num_joints * window * channels + 2 * num_joints
        self.fc1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        # 同样零初始化 fc_out: 训练 step 0 时所有 force token = 0 向量,
        # Gemma self-attention 对 force token 的 value 贡献为 0, 模型起步等价于"无力 baseline"
        # (=pi0.5 主干), 随训练梯度建立后逐渐让 force 进入注意力上下文.
        self.fc_out = nnx.Linear(
            hidden, embed_dim,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        # (arm_id, joint-within-arm-id) 嵌入. 拆成两个而不是单一 num_joints 嵌入,
        # 是因为同号关节 (例如 gripper) 在左右臂之间应该共享语义, arm side 只是 binary tag.
        # 小 std 初始化 (0.02 是 BERT/GPT 标准): step 0 时 force token 不再是严格 0,
        # 而是 (arm_emb + joint_emb) 的小幅扰动, 携带关节身份信息. fc_out 仍零初始化,
        # 故 MLP 动态分量从 0 起步, ID 偏置则一开始就在.
        if self.use_id_embedding:
            self.arm_embed = nnx.Embed(
                num_embeddings=num_arms,
                features=embed_dim,
                embedding_init=nnx.initializers.normal(stddev=0.02),
                rngs=rngs,
            )
            self.joint_embed = nnx.Embed(
                num_embeddings=self.per_arm_joints,
                features=embed_dim,
                embedding_init=nnx.initializers.normal(stddev=0.02),
                rngs=rngs,
            )

    def __call__(
        self,
        force_history: jax.Array,
        tau_hat_t: jax.Array,
        mismatch_t: jax.Array,
    ) -> jax.Array:

        B, K, W, C = force_history.shape
        if self.token_mode == "single":
            # 所有关节的历史 + prior extras 拼成一个向量 → 1 个 token.
            # 与 TA-VLA EXPERT_HIS_C 的单 token 聚合同构, 但在 encoder 侧且带 prior.
            hist = force_history.reshape(B, K * W * C)
            extras = jnp.concatenate([tau_hat_t, mismatch_t], axis=-1)  # (B, 2K)
            x = jnp.concatenate([hist, extras], axis=-1)                # (B, K*W*C + 2K)
            x = nnx.swish(self.fc1(x))
            x = nnx.swish(self.fc2(x))
            out = self.fc_out(x)                                        # (B, embed_dim)
            return out[:, None, :]                                      # (B, 1, embed_dim)

        # perjoint: 每个关节一个 token
        x = force_history.reshape(B, K, W * C)
        # 注入两个 prior 派生标量作为额外通道
        extras = jnp.stack([tau_hat_t, mismatch_t], axis=-1)  # (B, K, 2)
        x = jnp.concatenate([x, extras], axis=-1)             # (B, K, W*C + 2)
        # 共享 MLP 应用到每个关节（K 维当 batch）。
        x = nnx.swish(self.fc1(x))
        x = nnx.swish(self.fc2(x))
        out = self.fc_out(x)                                  # (B, K, embed_dim)
        if self.use_id_embedding:
            # K = num_arms * per_arm_joints, layout: [L_j1..L_j7, R_j1..R_j7]
            arm_ids = jnp.repeat(jnp.arange(self.num_arms), self.per_arm_joints)
            joint_ids = jnp.tile(jnp.arange(self.per_arm_joints), self.num_arms)
            out = out + self.arm_embed(arm_ids)[None] + self.joint_embed(joint_ids)[None]
        return out

class LongVLA(_model.BaseModel):
    # Force-as-Context VLA, 基于 pi0.5 主干.
    def __init__(self, config: LongVLAConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # PaliGemma 主干（与 pi0.5 一致）：
        # 第一个专家处理 prefix（视觉/语言/力），第二个专家处理 action suffix。
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # action expert 输入/输出投影
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        # 时间嵌入 MLP（pi0.5 的 AdaRMS 条件输入）
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        # 力先验。默认不喂视觉以抗 prior collapse；ablation 时可通过
        # config.prior_uses_vision=True 切换到含视觉版本做对照。
        # disable_force_prior=True 时彻底跳过此模块 (A2 ablation), 节省参数 + forward.
        if not config.disable_force_prior:
            self.force_prior = ForcePrior(
                lang_dim=paligemma_config.width,
                num_joints=config.num_joints,
                hidden=config.force_prior_hidden,
                vision_dim=paligemma_config.width if config.prior_uses_vision else None,
                rngs=rngs,
            )
        # 逐关节力 token 化器（输出维度 = paligemma width，便于直接拼进 prefix）
        self.joint_force_tokenizer = PerJointForceTokenizer(
            num_joints=config.num_joints,
            window=config.force_window,
            channels=config.force_channels,
            embed_dim=paligemma_config.width,
            hidden=config.joint_token_hidden,
            num_arms=config.num_arms,
            use_id_embedding=config.use_joint_id_embedding,
            token_mode=config.force_token_mode,
            rngs=rngs,
        )

        self.deterministic = True

    def _unpack_state(self, obs: _model.Observation) -> tuple[jax.Array, jax.Array]:

        if obs.force_history is None:
            raise ValueError(
                "obs.force_history is None — LongVLAInputs transform must populate it "
                "from the observation/qpos, observation/qvel, observation/filtered_effort fields."
            )
        return obs.state, obs.force_history

    def _compute_force_branch(
        self, obs: _model.Observation, vision_tokens: jax.Array, lang_tokens: jax.Array, lang_mask: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:

        # force_tokens  进入 prefix
        # tau_hat       ForcePrior 预测的期望力
        # tau_t      当前帧实测 τ_filtered，供辅助损失对比

        # 取出 force history，并提取当前帧 qpos / 实测力
        _, force_history = self._unpack_state(obs)
        qpos_t = force_history[:, :, -1, 0]   # 通道 0 = qpos
        tau_t = force_history[:, :, -1, 2]    # 通道 2 = τ_filtered

        # 语言池化（始终需要）；视觉池化（仅在 config.prior_uses_vision=True 时使用）。
        # 这里复用 prefix 阶段已经算好的 vision/lang tokens——既是 prior 的条件，也是
        # prefix 的输入，不需要重复跑 backbone，无双倍计算开销。
        #
        # lang_pool 必须做 masked mean: tokenized_prompt 被 padding 到 max_token_len=200,
        # PaliGemma 的 pad token embedding 非零, 直接 mean(axis=1) 会把 pad embedding 平均
        # 进去, 引入 prompt 长度依赖的偏置 (200 维序列里可能只有 ~10 个是真 prompt).
        cfg = self.config
        if cfg.disable_force_prior:
            # A2 ablation: 不调用 ForcePrior, tau_hat 强制为零向量.
            # 等价于 fc_out 零初始化后从未更新的状态, 但省掉 forward 开销且让结构上
            # 真的没有 prior 这条分支可学.
            tau_hat = jnp.zeros_like(tau_t)
        else:
            mask = lang_mask[..., None].astype(lang_tokens.dtype)               # (B, S, 1)
            lang_pool = (lang_tokens * mask).sum(axis=1) / jnp.clip(mask.sum(axis=1), 1.0)
            if cfg.prior_uses_vision:
                # vision token 内部不带 padding (每张图像产出固定数 token, 缺失图像由 image_mask
                # 在 attention 层面处理), 这里直接 mean 即可.
                vision_pool = vision_tokens.mean(axis=1)
                tau_hat = self.force_prior(lang_pool=lang_pool, qpos_t=qpos_t, vision_pool=vision_pool)
            else:
                tau_hat = self.force_prior(lang_pool=lang_pool, qpos_t=qpos_t)

        # stop_gradient: ForcePrior 只通过自己的 aux MSE loss 训练。如果让 action loss
        # 反向传播到 prior，模型会把 prior 调教成"让 mismatch 失效"的形态（接触越频繁
        # 越严重）。下游 token 看到的是 detached 版本，prior 只能向 τ̂ 拟合 τ 的方向更新。
        tau_hat_for_token = jax.lax.stop_gradient(tau_hat)
        mismatch_t = jnp.abs(tau_t - tau_hat_for_token)

        # 逐关节 token：raw (qpos, qvel, τ) 历史 + (τ̂_j, |τ_t-τ̂_j|) 两个 prior 派生标量
        force_tokens = self.joint_force_tokenizer(force_history, tau_hat_for_token, mismatch_t)
        return force_tokens, tau_hat, tau_t



    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        # """构建 prefix = [vision tokens, language tokens, force tokens].
        # 额外返回 (tau_hat, tau_t) 以便 compute_loss 计算辅助损失。
        
        # 视觉部分
        image_token_list = []
        image_mask_list = []
        for name in obs.images:
            tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            image_token_list.append(tokens)
            image_mask_list.append(einops.repeat(obs.image_masks[name], "b -> b s", s=tokens.shape[1]))
        vision_tokens = jnp.concatenate(image_token_list, axis=1)
        vision_mask = jnp.concatenate(image_mask_list, axis=1)
        n_vision = vision_tokens.shape[1]

        # 语言部分
        if obs.tokenized_prompt is not None:
            lang_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            lang_mask = obs.tokenized_prompt_mask
        else:
            # 兜底：无 prompt 时用一个空序列（基本不会触发）。
            B = vision_tokens.shape[0]
            lang_tokens = jnp.zeros((B, 0, vision_tokens.shape[-1]), dtype=vision_tokens.dtype)
            lang_mask = jnp.zeros((B, 0), dtype=jnp.bool_)
        n_lang = lang_tokens.shape[1]

        # 力部分
        force_tokens, tau_hat, tau_t = self._compute_force_branch(obs, vision_tokens, lang_tokens, lang_mask)
        n_force = force_tokens.shape[1]
        force_mask = jnp.ones(force_tokens.shape[:2], dtype=jnp.bool_)

        # 拼接
        tokens = jnp.concatenate([vision_tokens, lang_tokens, force_tokens], axis=1)
        input_mask = jnp.concatenate([vision_mask, lang_mask, force_mask], axis=1)
        # prefix 全部双向（不加因果掩码）；force token 也是双向，与 vision/lang 同等。
        ar_mask = jnp.zeros((n_vision + n_lang + n_force,), dtype=jnp.bool_)

        return tokens, input_mask, ar_mask, tau_hat, tau_t

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
    ]:
        # """suffix 仅由 noisy action token + AdaRMS 时间条件构成.

        # 相对原 scaffold 的关键变化：suffix 不再注入任何 force / interaction 信号。
        # 所有力的处理都在 prefix 阶段完成；action expert 通过 cross-attention 看到的是已经被 (V, L, F) 三向融合过的 prefix 上下文。
        
        # 时间嵌入 → AdaRMS 条件
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)

        action_tokens = self.action_in_proj(noisy_actions)
        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        # action expert 内部仅第一个 token 是因果起点，后续可全连接（与 pi0.5 一致）。
        ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1))
        return action_tokens, input_mask, ar_mask, time_emb

    #训练推理部分
    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # 薄包装: 复用 compute_loss_with_metrics 的前向, 丢弃辅助 metrics.
        # 保留 BaseModel 接口签名不变, 避免其他模型 / 训练脚本耦合到 metrics 字段.
        loss, _ = self.compute_loss_with_metrics(rng, observation, actions, train=train)
        return loss

    def compute_loss_with_metrics(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, jax.Array]]:
        # 返回 (chunked_loss, aux_metrics). aux 里目前包含分项 loss / tau 量级, 供训练循环
        # 通过 has_aux=True 拿到, log 到 wandb. 这样可以独立观察 ForcePrior 是否在学习,
        # 以及当前 aux_anomaly_weight=0.01 是不是合适的量级.
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, image_keys=LONGVLA_IMAGE_KEYS
        )

        batch_shape = actions.shape[:-2]
        # flow matching: x_t = t·noise + (1-t)·actions, 学预测 v_t = noise - actions
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 单次前向：prefix（含 force tokens）→ suffix（仅 action tokens）
        prefix_tokens, prefix_mask, prefix_ar_mask, tau_hat, tau_t = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # 主损失：flow matching MSE，每个 (batch, action_step) 上一个标量
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (B, ah)

        # 辅助损失：force prior 自监督。每个 batch 一个标量，广播到 (B, ah) 与主损失同形相加，
        # 让外层训练循环按相同规则求平均。
        anomaly_loss = jnp.mean(jnp.square(tau_t - tau_hat), axis=-1)  # (B,)
        anomaly_loss_b_ah = jnp.broadcast_to(anomaly_loss[:, None], flow_loss.shape)

        # A2 ablation: disable_force_prior=True 时 anomaly_loss = mean(tau_t^2), 没有学习意义,
        # 强制权重为 0 不进 total_loss. 仍保留在 aux metrics 里供 CSV 观察 tau_t 量级.
        aux_weight = 0.0 if self.config.disable_force_prior else self.config.aux_anomaly_weight
        total_loss = flow_loss + aux_weight * anomaly_loss_b_ah

        # aux metrics: 不参与 grad (仍走主 loss), 只是 jax 标量供 jit 内 device_get -> wandb.
        # 注意: 因为 stop_gradient 截断了 tau_hat 对 flow_loss 的反传, ForcePrior 唯一的梯度
        # 源就是 aux_anomaly_weight * anomaly_loss; anomaly_mse 直接反映这个学习信号大小.
        aux = {
            "loss/flow": jnp.mean(flow_loss),
            "loss/anomaly_mse": jnp.mean(anomaly_loss),
            "loss/anomaly_weighted": aux_weight * jnp.mean(anomaly_loss),
            # τ 量级: ForcePrior 没归一化 (force_history 不在 norm_stats 里), 直接看原始尺度.
            "force/tau_t_abs_mean": jnp.mean(jnp.abs(tau_t)),
            "force/tau_hat_abs_mean": jnp.mean(jnp.abs(tau_hat)),
            "force/mismatch_abs_mean": jnp.mean(jnp.abs(tau_t - tau_hat)),
        }
        return total_loss, aux

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=LONGVLA_IMAGE_KEYS
        )
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 推理时先把 prefix 跑一遍并缓存 KV，后续每步只跑 suffix。
        # tau_hat / tau_t 在推理阶段不需要，丢弃即可。
        prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (_, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions, adarms_cond=[None, None]
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_b = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_b, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
