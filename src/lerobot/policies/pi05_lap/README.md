# PI05 LAP

从当前 `pi05/` 复制并扩展的两阶段策略，原 PI05 策略保持独立。
输入仍然是 PI05 的当前图像、任务指令和离散化状态，使用原有 processor 和归一化流程。

## 配置

仅新增三个配置字段：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `policy.training_stage` | `post-training` | `pre-training` 或 `post-training` |
| `policy.lam_model_path` | `null` | UniVLA stage-2 LAM 的本地 `.ckpt` 文件，仅预训练需要 |
| `policy.lam_camera_key` | `null` | 单个 RGB 相机全名或短名，省略时自动选主视角 |

自动选择相机时依次优先 `image`、`base_0_rgb`、`primary`、`front`、
`agentview_image`、`image_primary`，否则使用 metadata 中的第一个 RGB 相机。
不接受相机列表；显式填写的相机不存在或不是 RGB 时直接报错。

## 预训练

必须从原 PI05 checkpoint 初始化，也可从已有 PI05 LAP checkpoint 继续。
`--policy.type=pi05_lap --policy.path=<PI05目录>` 会继承源 checkpoint 的模型配置，
再应用命令行覆盖；源目录需有 LeRobot 的 `config.json`、`model.safetensors` 和 processor 文件。

对每个样本取所选相机的 `(t, min(t+H, episode_last))`，其中 **H = `policy.chunk_size`**。
动作 chunk 保留原生 PI05 的采样、末尾重复最后动作、归一化及 padding 行为，不另做补零。
两个 teacher 图像按参考实现使用 RGB PIL bilinear resize 到 224×224，之后由 UniVLA
做 ImageNet 归一化。teacher 使用未增强图像；VLM 仍可按原配置增强当前图像。
预训练使用 map-style LeRobot 数据集，不支持 streaming。

UniVLA 固定使用之前的 stage-2 `vq_action`：4 个 token、16 项码本、latent dimension 128。
所有必要推理源码已纳入 `lam/`，不依赖外部 UniVLA 工程，也不会下载 DINOv2 权重。
LAM 在每个 CUDA 进程上以冻结 FP32 模式在线生成 codebook 索引 `0..15`，独立于学生 autocast，
不参与 DDP、优化器或策略 checkpoint。

学生 tokenizer 在内存中加入 `<latent_action_start>` 及 `<latent_action_0>` 到
`<latent_action_15>`，扩展 VLM embedding 和 LM head。实际 token ID 由 tokenizer 分配，
不会把码本索引当成原词表 ID，也不修改原始 tokenizer 目录。
训练时 teacher forcing 输入 `[START, y0, y1, y2]`，预测 `[y0, y1, y2, y3]`。
当前观测 prefix 保持 PI05 的双向注意力，latent suffix 使用因果注意力，交叉熵只计算这四个位置。
冻结动作专家、动作投影和时间 MLP；训练 VLM 与 LM head，视觉编码器是否冻结沿用
`freeze_vision_encoder`。`train_expert_only` 必须为 `false`。

checkpoint 同时保存 `latent_tokenizer/` 和 `latent_action_tokens.json`，后者记录准确的 ID 映射。
加载 LAP checkpoint 时验证两者一致，严格恢复已训练 embedding、LM head 和其他权重。
原始 PI05 权重加载时仅初始化新增词表行；加载失败直接报错。

## 动作微调

将启动脚本中的 `policy.path` 改为预训练输出的 `pretrained_model` 目录，
`policy.training_stage` 改为 `post-training`，并更换 `output_dir`。
这一阶段只使用真实动作的原生 PI05 flow-matching 损失，不加载 LAM，也不采样未来图像。
动作专家从第一阶段保留的 PI05 权重开始训练，VLM 从预训练结果继续；
`train_expert_only`、`freeze_vision_encoder` 等现有选项仍然有效。

动作推理使用原有 `select_action` / `predict_action_chunk`。
预训练模型可用 `predict_latent_actions(processed_batch)` 从当前观测自回归生成四个码本索引；
它不需要未来图像或 LAM 权重。动作推理需切到 `post-training`。

## 启动

安装依赖：`uv sync --extra training --extra pi05-lap`。
仓库根目录的 `train_pi05_lap.sh` 为直接填写路径的单命令模板，默认 `H=12`。
修改原 PI05、LAM、数据集和输出目录后运行：

```bash
sh train_pi05_lap.sh
```

服务器 CUDA 训练和实际 checkpoint 的效果需在目标服务器验证。
