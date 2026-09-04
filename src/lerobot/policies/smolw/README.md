# SmolW

SmolW 直接以仓库中的原始 SmolVLA 为基线，不依赖 `smol_actionmem`，当前只支持
LeRobot 格式数据。

## 时序定义

对当前时刻 `t`，`motion_horizon=H`、`memory_stride=s`：

- 历史窗口：`[t-(H-1)s, ..., t-s, t]`，共 `H` 帧；
- 未来窗口：`[t+1, ..., t+H]`，共 `H` 帧；
- action chunk：`[a_t, ..., a_{t+H-1}]`。

这样未来窗口的最后一帧 `o_{t+H}` 正好是执行完 `H` 个 action 后的观测。VidTwin
内部始终使用固定 16 帧：两个窗口都会通过 `linspace` 均匀采样为 16 帧。因此 action
horizon `H` 可以独立配置，而 VidTwin motion 和 z token 数始终固定为 16。

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
expert、action 投影和 z flow heads。视觉编码器是否冻结由
`freeze_vision_encoder` 控制。

### `action_only`

模拟 future-motion 预测完全正确的情况：VidTwin 从真实未来窗口提取 1792 维 latent，
将它转换成 16 个逐时间步 z latent，作为 flow matching 的 teacher target。VLM、
`M_t`/motion projector 和 future-motion head 全部冻结；训练 action expert、SmolVLA
action/time 投影和 z flow heads。

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

VidTwin 的两个 `[B,8,16,7]` motion latent 沿通道拼接后按原 CoWVLA 顺序得到
`[B,16,7,16]`。每个 temporal slot 的 `7*16=112` 维特征独立做无可训练参数的
LayerNorm，直接形成固定目标 `[B,16,112]`，不会用 Linear 混合时间位置，也不会让
target space 随训练漂移。noisy z 在进入 action expert 前才通过 `112→expert_hidden_size`
的输入投影，expert 输出则通过 `expert_hidden_size→112` 回到 z velocity 空间。

flow matching 内部 suffix 按 `[z_1,...,z_16,a_1,...,a_H]` 分块排列：

- action-to-action 子矩阵保持原始 SmolVLA 因果注意力，`a_i` 可读取 `a_1...a_i`；
- warmup 结束后，每个 `a_i` 都能读取全部 `z_1...z_16`，因此完整 predicted motion
  指导整个 action chunk；
- 16 个 z token 互相可见，但任何 `z_i` 都不能读取 action，避免从 GT/noisy action 泄漏标签；
- z 能读取包含 `M_t` 的完整 prefix，action 不直接读取 `M_t`，而是经 z 获得 motion 信息；
- action 和 z 使用同一个 flow timestep，但分别采样噪声并分别预测 velocity；
- 推理时联合去噪 16 个 z 和 H 个 action，策略最终只返回 action chunk。

`z_condition_warmup_steps=m` 控制 action 条件课程学习：前 `m` 次 action 训练 forward
中仅屏蔽 `action→z` 注意力，action 按原始 SmolVLA 路径学习，而 z flow loss 仍正常
训练；从第 `m+1` 次开始恢复所有 `action→z` 边。计数器保存在 policy checkpoint 中，
恢复训练不会重新开始 warmup；eval/推理始终启用 z condition。若设为 `0`，从第一步
起就使用 z。

当前 RTC 不支持这种联合状态去噪，启用时会明确报错。

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
Z_CONDITION_WARMUP_STEPS=10000 \
OUTPUT_DIR=/path/to/smolw-stage2 \
HORIZON=16 MEMORY_STRIDE=1 \
bash train_smolw_lr.sh
```

共同训练：

```bash
TRAIN_MODE=jointly \
Z_CONDITION_WARMUP_STEPS=10000 \
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
TensorBoard 还会记录 `z_condition_step` 和二值的 `z_condition_active`，便于确认课程切换。
