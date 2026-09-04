# SmolW

SmolW 直接以仓库中的原始 SmolVLA 为基线，不依赖 `smol_actionmem`，当前只支持
LeRobot 格式数据。模型不再包含独立的 VLM future-motion regression head，也没有
`train_mode`；训练目标始终是联合 flow matching 生成 `(z, action)`。

## 时序定义

对当前时刻 `t`，`motion_horizon=H`、`memory_stride=s`：

- 历史窗口：`[t-(H-1)s, ..., t-s, t]`，共 `H` 帧；
- 未来窗口：`[t+1, ..., t+H]`，共 `H` 帧；
- action chunk：`[a_t, ..., a_{t+H-1}]`。

VLM 的普通视觉、语言和状态输入仍只使用当前观测 `o_t`。历史窗口由冻结的 VidTwin
编码并通过 prefix 末尾的 `M_t` query 提供条件。未来窗口只在训练时由冻结的 VidTwin
编码，用来构造 GT z target；推理不需要未来图像。

VidTwin 内部始终使用固定 16 帧。若 `H != 16`，历史和未来窗口都会按照 CoWVLA 的
`linspace` 规则均匀采样为 16 帧。因此 action horizon `H` 可配置，而 z token 数固定为
16。

## 联合 `(z, action)` flow matching

VidTwin 的两个 `[B,8,16,7]` motion latent 沿通道拼接后按 CoWVLA 顺序得到
`[B,16,7,16]`。每个 temporal slot 的 `7*16=112` 维特征独立执行无可训练参数的
LayerNorm，形成固定的 GT target：

```text
z_target: [B, 16, 112]
```

z 和 action 使用同一个 flow timestep、独立高斯噪声：

```text
z_t = t * z_noise + (1-t) * z_target
a_t = t * a_noise + (1-t) * action

u_z = z_noise - z_target
u_a = a_noise - action
```

noisy z 逐 token 通过 `112→expert_hidden_size` 输入投影，action 使用原始 SmolVLA
action projection。action expert 同时输出两组 hidden states，再分别投影成 z velocity
和 action velocity：

```text
loss = action_flow_loss + z_loss_weight * z_flow_loss
```

suffix 固定排列为 `[z_1,...,z_16,a_1,...,a_H]`：

- 16 个 z token 互相可见，但不能读取 action；
- 每个 action token 都能读取全部 16 个 z；
- action-to-action 子矩阵保持原始 SmolVLA 因果注意力；
- z 能读取包含 `M_t` 的完整 prefix；action 不直接读取 `M_t`，只通过 z 获得历史
  motion 信息。

训练时 VLM 条件路径、`M_t`/past-motion projector、action expert、action heads 和 z heads
共同接受 flow loss 的梯度；视觉编码器是否冻结由 `freeze_vision_encoder` 控制。
`state_proj` 遵循 SmolVLA 的 `train_state_proj` 开关。当前 RTC 不支持联合 `(z, action)`
去噪。

推理从 `z_noise` 和 `action_noise` 出发，使用相同的 action expert 联合积分。最终策略只
返回 action chunk，内部生成的 `[B,16,112]` z 不会被解码成图像。

## 准备 base

模型结构已变化，不兼容旧 SmolW checkpoint。请从原始 SmolVLA 重新生成 base：

```bash
SOURCE_SMOLVLA_DIR=/path/to/smolvla \
SMOLW_MODEL_DIR=/path/to/smolw-base \
VIDTWIN_CHECKPOINT_PATH=/path/to/vidtwin.ckpt \
HORIZON=16 MEMORY_STRIDE=1 \
bash convert_smolw_base.sh
```

外部只需要 VidTwin `.ckpt`；网络源码和
`vidtwin_structure_7_7_8_dynamics_7_8.yaml` 已放在 SmolW 的 `vidtwin/` 子目录，运行时
不会导入 `scripts/CoWVLA`。VidTwin 是冻结的惰性模块，其权重不会重复写入 SmolW
checkpoint。

首次使用时安装附加依赖：

```bash
uv sync --extra smolw --extra training
```

## 启动训练

```bash
OUTPUT_DIR=/path/to/smolw \
HORIZON=16 MEMORY_STRIDE=1 \
bash train_smolw_lr.sh
```

训练脚本设置 `drop_n_last_frames=H`，确保未来 GT z 窗口不会跨过 episode 末尾。episode
开头缺少的历史帧沿用 LeRobot 的边界补帧，推理时也会重复最早可用帧填满历史。

TensorBoard 默认每 10 step 写一次 scalar，日志目录为 `${OUTPUT_DIR}/tensorboard`：

```bash
tensorboard --logdir /path/to/output/tensorboard
```

主要指标为 `action_flow_loss`、`z_flow_loss`、`weighted_z_flow_loss`、`z_target_rms` 和
总 `loss`。
