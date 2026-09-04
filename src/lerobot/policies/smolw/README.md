# SmolW

SmolW 直接以仓库中的原始 SmolVLA 为基线，不依赖 `smol_actionmem`，当前只支持
LeRobot 格式数据。

## 时序定义

对当前时刻 `t`，`motion_horizon=H`、`memory_stride=s`：

- 历史窗口：`[t-(H-1)s, ..., t-s, t]`，共 `H` 帧；
- 未来窗口：`[t+1, ..., t+H]`，共 `H` 帧；
- action chunk：`[a_t, ..., a_{t+H-1}]`。

这样未来窗口的最后一帧 `o_{t+H}` 正好是执行完 `H` 个 action 后的观测。VidTwin
内部始终使用固定 16 帧：两个窗口都会通过 `linspace` 均匀采样为 16 帧。因此当
`H=16` 时逐帧输入；`H>16` 时下采样，`H<16` 时会重复部分帧。

VLM 的普通视觉、语言和状态输入仍只使用当前观测 `o_t`。历史窗口由冻结的 VidTwin
编码为 latent motion，并通过追加在 VLM prefix 末尾的 `M_t` query 输入模型。

## 三种训练模式

训练目标统一由 `train_mode` 切换。

### `motion_only`

不准备 action、不添加 flow noise，也不运行 action expert。VLM 根据当前观测和历史
VidTwin latent，回归未来窗口的 1792 维 VidTwin latent：

```text
loss = motion_loss_weight * SmoothL1(pred_motion, target_motion)
```

本模式训练 VLM 的有效层、`M_t`/motion projector 和 future-motion head；冻结 action
expert、action 投影和 motion-to-action condition。视觉编码器是否冻结由
`freeze_vision_encoder` 控制。

### `action_only`

模拟 future-motion 预测完全正确的情况：VidTwin 从真实未来窗口提取 1792 维 latent，
将它转换成一个 horizon 级 z latent，作为 flow matching 的 teacher target。VLM、
`M_t`/motion projector 和 future-motion head 全部冻结；训练 action expert、SmolVLA
action/time 投影、motion-to-z projection 和 z flow heads。

### `jointly`

同时计算 motion regression 和 `(z, action)` flow-matching loss。z target 由 VLM 自己预测
的 future motion 转换而来，而不使用真实 future motion，因此训练路径和实际推理路径一致：

```text
loss = action_flow_loss
     + z_loss_weight * z_flow_loss
     + motion_loss_weight * SmoothL1(pred_motion, target_motion)
```

VLM/motion 分支和 action expert 分支同时训练。若 `detach_motion_condition=false`，action
expert 的 z loss 也会经过 z target 回传到 VLM/motion 分支。

## `(z, action)` suffix 与注意力

flow matching 不再把 predicted motion 直接加到每个 action token 上。内部 suffix 按
`[a_1, ..., a_H, z]` 排列，其中 z 是一个代表完整 horizon 的全局 latent token，连续
维度等于 action expert hidden size：

- H 个 action token 的注意力与原始 SmolVLA 完全相同，不能读取 `M_t` 或 z；
- z token 可以读取普通 VLM prefix、`M_t`、全部 H 个 action token 和自身；
- action 和 z 使用同一个 flow timestep，但分别采样噪声并分别预测 velocity；
- 推理时联合去噪 `(z, action)`，策略最终只返回 action chunk。

因此 motion 不再作为 action token 的显式加法 condition，而是通过 horizon-level z
辅助目标与 action expert 对齐。当前 RTC 不支持这种联合状态去噪，启用时会明确报错。

所有可训练参数均使用 SmolVLA 配置中的同一个 `optimizer_lr`，当前没有模块级学习率分组。

无论使用哪种模式，训练 batch 都需要未来窗口：`motion_only`/`jointly` 用它生成 motion
监督，`action_only` 用它生成 oracle motion condition。实际仿真推理始终只能使用模型预测
的 future motion；因此只训练 `action_only` 的模型不能独立验证完整链路，通常需要加载
已经训练过 motion 分支的 checkpoint。

## 准备 base

先从原始 SmolVLA 生成一次 SmolW base artifact：

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

不要在 LeRobot 环境中安装 CoWVLA 的完整 requirements；其中固定的 Torch 和
Transformers 版本与当前 LeRobot 不一致。

## 启动训练

只训练 motion：

```bash
TRAIN_MODE=motion_only \
OUTPUT_DIR=/path/to/smolw-stage1 \
HORIZON=16 MEMORY_STRIDE=1 \
bash train_smolw_lr.sh
```

使用真实 future motion 只训练 action：

```bash
TRAIN_MODE=action_only \
OUTPUT_DIR=/path/to/smolw-stage2 \
HORIZON=16 MEMORY_STRIDE=1 \
bash train_smolw_lr.sh
```

共同训练：

```bash
TRAIN_MODE=jointly \
OUTPUT_DIR=/path/to/smolw-joint \
HORIZON=16 MEMORY_STRIDE=1 \
bash train_smolw_lr.sh
```

模型输入/输出目录和 checkpoint 路径由训练启动脚本中的参数控制。三个模式都会设置
`drop_n_last_frames=H`，避免未来窗口跨过 episode 末尾。episode 开头缺少的历史帧沿用
LeRobot 的边界补帧，推理时也会重复最早可用帧填满历史。action chunk 长度固定为
`HORIZON`。

训练脚本默认每 10 step 写一次 TensorBoard scalar，日志目录为
`${OUTPUT_DIR}/tensorboard`：

```bash
tensorboard --logdir /path/to/output/tensorboard
```

可通过 `TENSORBOARD_LOG_DIR` 覆盖日志目录。
