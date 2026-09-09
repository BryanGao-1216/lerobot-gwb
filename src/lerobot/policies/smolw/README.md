# SmolW

SmolW 直接以仓库中的原始 SmolVLA 为基线，当前只支持
LeRobot 格式数据。通过启动脚本中的 `train_version` 选择增量实验；模型没有独立的
VLM future-motion regression head。

## A/B/C/D 增量实验

| train_version | 历史 motion | 未来 z 监督 | 动作读取生成 z |
| --- | --- | --- | --- |
| A | 不使用 | 不使用 | 不使用 |
| B | 追加 `M_t`，动作直接可读 | 不使用 | 不使用 |
| C | 同 B | 增加 `z_loss_weight * z_flow_loss` | 不读取 |
| D | 同 B | 同 C | 通过零初始化的门控残差 attention 读取 |

- **A** 直接实例化原始 `VLAFlowMatching`，复用 `SmolVLAPolicy` 的训练 loss、采样及动作队列；
  不创建 motion 模块、不加载 VidTwin，数据只请求当前观测 `[0]`。
- **B** 仅请求历史窗口，保留原动作 heads、时间步分布、因果 action attention 和训练方式。
  历史投影沿用现有的 `1792 → hidden → VLM hidden`，动作直接读取 `M_t`。
- **C** 训练时加入未来 16 个 z token 和现有 z flow loss。共享 expert 内 z/action 双向隔离，
  动作仍直接读取 `M_t`；z 监督通过共享参数起辅助作用。推理与 B 相同，不生成 z。
- **D** 在 C 的 expert 输出后增加独立的单头 cross-attention：action hidden 作 query，z hidden
  作 key/value，结果乘以 `tanh(gate)` 再残差加到 action hidden，最后使用原 action head。
  `gate` 是可学习标量，初始为 0，所以相同权重、输入和噪声下 D 初始动作路径等价于 C。
  z 不读取动作。门控独立归一化，不会向原 action attention 的 softmax 分母添加 z。

BCD 的动作位置均按照含 `M_t` 的有效 prefix 长度计算。C/D 的 z 使用相同起点的独立时间流；
追加 z 不再改变动作位置。BCD 训练均沿用原 SmolVLA 的 prefix/suffix 联合前向（不使用 KV
cache）；推理时缓存 prefix。A 的 action 训练/推理直接调用原实现。

**共同控制条件：** ABCD 均使用原 SmolVLA 的预处理与 `action_is_pad`，不合成静止尾部动作。
加载旧 SmolW base 的 processor 时会移除该合成步骤，避免把尾部监督变更混入实验。
未来视频的边界补帧仍由 LeRobot 处理。启动脚本的 `z_loss_weight=1.0`、horizon、学习率、
batch size、步数和其他数值参数保持不变；A/B 不计算 z loss，但也不改写其配置值。
因此 A 对齐的是**同一组启动超参数和原始权重下**的 SmolVLA，而不是另一个 horizon/步数的旧实验。

从同一个由原 SmolVLA 转换的 base 分别启动（默认 A）：

```bash
train_version=A bash train_smolw_lr.sh
train_version=B bash train_smolw_lr.sh
train_version=C bash train_smolw_lr.sh
train_version=D bash train_smolw_lr.sh
```

默认输出分别为 `smolw-union-A`、`smolw-union-B`、`smolw-union-C`、`smolw-union-D`，
也可显式设置 `OUTPUT_DIR`。不要用已训练的旧联合模型作为 A 的公平基线初始化。
`train_version` 会保存到 checkpoint 的 config，评估和恢复训练时自动沿用，通常不应再覆盖它。
读取旧 base 时 A/B 会忽略未使用的 motion/z 权重；D 新增的门控 attention 从初始化开始训练。

未包含 `train_version`（或值为 `null`）的旧 checkpoint 保持下面的旧联合模式，包括原注意力和
静止尾部处理，不会被自动解释成 A/B/C/D。`null` 仅用于旧 checkpoint 兼容，训练启动脚本只接受
A/B/C/D 四个值。

## 时序定义

对当前时刻 `t`，`motion_horizon=H`、`memory_stride=s`：

- 历史窗口：`[t-(H-1)s, ..., t-s, t]`，共 `H` 帧；
- 未来窗口：`[t+1, ..., t+H]`，共 `H` 帧；
- action chunk：`[a_t, ..., a_{t+H-1}]`。

除 A 外，VLM 的普通视觉、语言和状态输入仍只使用当前观测 `o_t`。历史窗口由冻结的 VidTwin
编码并通过 prefix 末尾的 `M_t` query 提供条件。未来窗口只在训练时由冻结的 VidTwin
编码，用来构造 C/D 或旧联合模式的 GT z target；B 不请求未来窗口，推理均不需要未来图像。

VidTwin 内部始终使用固定 16 帧。若 `H != 16`，历史和未来窗口都会按照 CoWVLA 的
`linspace` 规则均匀采样为 16 帧。因此 action horizon `H` 可配置，而 z token 数固定为
16。

## z 监督与旧联合模式

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

C/D 训练和旧联合模式的 suffix 排列为 `[z_1,...,z_16,a_1,...,a_H]`。
**以下 attention 规则仅描述未设置 train_version 的旧联合模式；C/D 使用上表的隔离与门控：**

- 16 个 z token 互相可见，但不能读取 action；
- 每个 action token 都能读取全部 16 个 z；
- action-to-action 子矩阵保持原始 SmolVLA 因果注意力；
- z 能读取包含 `M_t` 的完整 prefix；action 不直接读取 `M_t`，只通过 z 获得历史
  motion 信息。

训练时 VLM 条件路径、`M_t`/past-motion projector、action expert、action heads 和 z heads
共同接受 flow loss 的梯度；视觉编码器是否冻结由 `freeze_vision_encoder` 控制。
`state_proj` 遵循 SmolVLA 的 `train_state_proj` 开关。当前 RTC 不支持联合 `(z, action)`
去噪。

旧联合模式和 D 推理从 `z_noise` 和 `action_noise` 出发，使用相同的 action expert 联合积分。最终策略只
返回 action chunk，内部生成的 `[B,16,112]` z 不会被解码成图像。

## 准备 base

ABCD 可共用现有转换脚本生成的 SmolW base；如果尚未准备，请从原始 SmolVLA 生成：

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
train_version=A \
HORIZON=16 MEMORY_STRIDE=1 \
bash train_smolw_lr.sh
```

训练脚本设置 `drop_n_last_frames=0`，因此每个 episode 的所有帧都可以作为训练起点。
**仅旧联合模式**采用以下静止动作尾部监督；ABCD 始终保留原始 action padding mask。超出
episode 末尾的未来图像沿用 LeRobot 的边界补帧并重复最后一帧，使未来 GT z 表示到达终态
后保持静止。LIBERO 的合成尾部动作在归一化前构造为前 6 维相对位姿增量为 0、最后一维
保持最后一个有效夹爪命令；这些动作会取消 `action_is_pad` 并参与 action flow loss。episode
开头缺少的历史帧同样使用边界补帧，推理时也会重复最早可用帧填满历史。

TensorBoard 默认每 10 step 写一次 scalar，日志目录为 `${OUTPUT_DIR}/tensorboard`：

```bash
tensorboard --logdir /path/to/output/tensorboard
```

主要指标为 `action_flow_loss`、`z_flow_loss`、`weighted_z_flow_loss`、`z_target_rms` 和
总 `loss`。A 记录原 SmolVLA 指标及 `action_flow_loss` 别名；B 的 z loss 为 0，A/B 均不记录
`z_target_rms`。D 额外记录 `motion_condition_gate=tanh(gate)`。
