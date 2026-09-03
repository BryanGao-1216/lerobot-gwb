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

## 两阶段训练

### 第一阶段：`world_model`

第一阶段不准备 action、不添加 flow noise，也不运行 action expert，只训练以下两个能力：

1. VLM 根据当前观测和历史 VidTwin latent，预测未来窗口的 VidTwin latent；
2. 使用当前帧 visual tokens 和预测的 future motion，预测 `o_{t+H}` 的 visual tokens。

未来 visual token 的 teacher 是冻结的 SmolVLM `vision_model + connector`。预测头采用
“当前帧 tokens + future residual”的形式；decoder 只直接读取当前 tokens 和预测出的
future motion，不直接旁路读取真实 future latent，确保图像预测损失能够约束 motion
prediction。损失为：

```text
loss = motion_loss_weight * SmoothL1(pred_motion, target_motion)
     + future_visual_loss_weight * (
           SmoothL1(pred_visual_tokens, target_visual_tokens)
           + future_visual_cosine_weight * cosine_loss
       )
```

本阶段训练 VLM 的有效文本层、`M_t`/motion projector、future-motion head 和 visual-token
decoder；冻结 action expert、action 投影、motion-to-action condition，以及作为 teacher
的视觉编码器和 connector。

### 第二阶段：`action_expert_only`

第二阶段应从第一阶段 checkpoint 加载整个策略。VLM、motion prediction 和 visual-token
decoder 全部冻结；每个 batch 只读取历史窗口，不再读取未来图像或生成未来监督。
冻结的一阶段模型先预测 future motion，然后仅训练 action expert、SmolVLA action/time
投影和 `future_motion_condition_proj`，用原有 flow matching 输出 action chunk。

所有可训练参数均使用 SmolVLA 配置中的同一个 `optimizer_lr`，当前没有模块级学习率分组。
`joint` 模式保留用于联合训练或消融，但不是默认两阶段流程。

## 准备 base

先从原始 SmolVLA 生成一次 SmolW base artifact：

```bash
SOURCE_SMOLVLA_DIR=/path/to/smolvla \
SMOLW_MODEL_DIR=/path/to/smolw-base \
VIDTWIN_CHECKPOINT_PATH=/path/to/vidtwin.ckpt \
HORIZON=16 MEMORY_STRIDE=1 \
bash convert_smolw_base.sh
```

这次新增了 visual-token decoder 参数，因此旧版 SmolW base 不能直接作为第一阶段起点，
需要重新执行一次转换脚本。

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

第一阶段：

```bash
TRAINING_STAGE=world_model \
POLICY_PATH=/path/to/smolw-base \
OUTPUT_DIR=/path/to/smolw-stage1 \
HORIZON=16 MEMORY_STRIDE=1 \
bash train_smolw_lr.sh
```

第二阶段，把 `POLICY_PATH` 指向第一阶段保存的 `pretrained_model` 目录：

```bash
TRAINING_STAGE=action_expert_only \
POLICY_PATH=/path/to/smolw-stage1/checkpoints/<step>/pretrained_model \
OUTPUT_DIR=/path/to/smolw-stage2 \
HORIZON=16 MEMORY_STRIDE=1 \
bash train_smolw_lr.sh
```

启动脚本会为 `world_model`/`joint` 自动设置 `drop_n_last_frames=H`，为
`action_expert_only` 设置为 0；可用 `DROP_N_LAST_FRAMES` 覆盖。若不设置
`MOTION_CAMERA_KEY`，策略默认使用配置中的第一个视觉输入。episode 开头缺少的历史帧
沿用 LeRobot 的边界补帧，推理时也会重复最早可用帧填满历史。action chunk 长度固定为
`HORIZON`，推理时每次实际执行的步数可用 `N_ACTION_STEPS` 单独设置。

训练脚本默认每 10 step 写一次 TensorBoard scalar，日志目录为
`${OUTPUT_DIR}/tensorboard`：

```bash
tensorboard --logdir /path/to/output/tensorboard
```

可通过 `TENSORBOARD_ENABLE`、`TENSORBOARD_LOG_FREQ` 和 `TENSORBOARD_LOG_DIR` 覆盖。
