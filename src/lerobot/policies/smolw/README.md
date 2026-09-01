# SmolW

SmolW 直接以仓库中的原始 SmolVLA 为基线，不依赖 `smol_actionmem`。当前实现只接入
LeRobot 数据集。

训练时，LeRobot 为时刻 `t` 提供两组图像：

- 过去运动窗口：`[-(H-1)s, ..., -s, 0]`，共 `H` 帧；
- 未来运动窗口：`[0, 1, ..., H-1]`，共 `H` 帧。

VLM 的普通视觉、语言和状态输入仍只使用 `t` 时刻。过去窗口经过冻结的 VidTwin
后，作为一个追加在 VLM prefix 末尾的 `M_t` query；该 query 的输出回归未来窗口的
VidTwin motion latent。预测 latent 再投影到 action expert hidden size，并加到每个
SmolVLA flow-matching action token 上。action expert 的 attention 会屏蔽 `M_t` 本身，
保证它获得过去运动信息的唯一路径是“VLM 预测出的未来 latent”。

VidTwin 提取逻辑保持 CoWVLA 的关键约定：使用 `linspace` 采样固定 16 帧、Resize +
CenterCrop 到 224、归一化到 `[-1, 1]`、调用 `encode`、拼接 `z_motion_x/y`，最后按
`b d f n -> b (f n d)` 展平。VidTwin 是冻结的惰性外部模块，不写入 SmolW checkpoint。

先从原始 SmolVLA 生成一次 SmolW base artifact：

```bash
SOURCE_SMOLVLA_DIR=/path/to/smolvla \
SMOLW_MODEL_DIR=/path/to/smolw-base \
VIDTWIN_REPO_DIR=/path/to/CoWVLA \
VIDTWIN_CHECKPOINT_PATH=/path/to/vidtwin.ckpt \
HORIZON=20 MEMORY_STRIDE=1 \
bash convert_smolw_base.sh
```

然后训练；模型、VidTwin、数据集及输出目录均由启动脚本变量控制：

```bash
SMOLW_MODEL_DIR=/path/to/smolw-base \
VIDTWIN_REPO_DIR=/path/to/CoWVLA \
VIDTWIN_CHECKPOINT_PATH=/path/to/vidtwin.ckpt \
DATASET_ROOT=/path/to/lerobot_dataset \
OUTPUT_DIR=/path/to/output \
HORIZON=20 MEMORY_STRIDE=1 \
bash train_smolw_lr.sh
```

若不设置 `MOTION_CAMERA_KEY`，默认使用策略配置中的第一个视觉输入。训练会自动丢弃
每个 episode 最后的 `H-1` 个采样起点，避免 future motion target 含补帧；episode
开头缺少的历史帧则沿用 LeRobot 的边界补帧。推理启动阶段同样重复最早可用帧填满历史。
