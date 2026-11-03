# PI05 LAP

多数据集混合训练可使用 `--dataset.config_path=dataset/config.json`，各数据集独立配置字段映射、相机和抽样权重。详见 [数据配置说明](../../../../docs/dataset_config.md)。

JSON 中每个数据集的 `camera_fields` 把 VLM 视角映射到数据集原始相机字段，可以选择部分相机，并用 `null`、`""` 或省略槽位表示空视角。`lam_camera_key` 填写其中一个非空原始相机字段，用于 latent 预训练；不能选择 VLM 输入之外的相机。只读取实际选中的相机。

从当前 `pi05/` 扩展的两阶段策略：阶段一学习四个 latent action token，阶段二使用 PI05
flow matching 学习真实动作。PI05 与 LAP 共用对齐 OpenPI 的输入输出实现。

## OpenPI 输入输出对齐

参考固定为 [OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)，
图像训练增强采用其 PyTorch 实现。仓库中的 baseline 和阶段二脚本对齐官方 `pi05_libero`
配方的输入输出；动作 horizon、学习率、有效 batch、训练步数与留出评估按训练入口配置。

| 项目 | 当前实现 |
| --- | --- |
| 文本 | 官方 PaliGemma SentencePiece；清理下划线和换行；BOS、单独编码的末尾换行、截断和 padding 与官方相同 |
| 状态 | LAP 默认 `discrete_state_input=false`，与 `pi05_libero` 相同；启用时先归一化实际状态维度，再按官方 256 bins 写入 prompt，补齐维度不写入文本 |
| 图像槽位 | 固定顺序 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`；LIBERO 使用前两路，第三路零图像且 mask=false |
| 图像数值 | 浮点图像先按官方方式转 uint8，PIL bilinear 等比例 resize/居中补零，随后转换到 `[-1,1]` |
| 图像增强 | 仅训练时使用官方 PyTorch 裁剪、旋转和颜色增强；wrist 不做几何增强；验证和推理关闭；LAM teacher 始终使用未增强图像 |
| 状态/动作归一化 | `(x-q01)/(q99-q01+1e-6)*2-1`，反归一化也保留相同 epsilon 和计算顺序 |
| 动作训练输出 | 实际动作先归一化再补零到模型的 32 维，flow matching 对全部 32 维取平均，包括补齐维度 |
| 动作推理输出 | 模型产生 32 维动作，截取真实动作维度并反归一化；LIBERO 输出 `[B,H,7]`，其中 H 为入口配置的 `policy.chunk_size` |
| 扩展词表 | 保留 Gemma embedding 类型及 `sqrt(hidden_size)` 缩放，原词表权重和原 token 表示不变；输入 embedding 与 LM head 按模型配置共享权重 |

普通 `PI05Config` 的 `discrete_state_input` 仍按官方通用 PI05 默认为 `true`；LIBERO 应明确设为
`false`。动作步数不跟随官方配方强制设为 10：`policy.chunk_size` 决定训练和推理的 horizon，
`policy.n_action_steps` 决定一次执行多少步（不能超过 chunk_size）。两者由训练入口的参数或
配置决定；未显式覆盖时沿用源 checkpoint 的配置，通用配置默认均为 50。
模型 action dim=32，LIBERO 原始 state/action 分别为 8/7 维。
`dataset/config.json` 明确映射主相机和腕部相机，关闭额外的数据集增强及 ImageNet 归一化。
自定义数据的相机含义、动作坐标系、单位和统计量仍必须由实际数据正确提供。

首次使用会缓存官方公开的 `paligemma_tokenizer.model`，也可用
`--policy.tokenizer_model_path=/path/to/paligemma_tokenizer.model` 指定文件。
所有路径均校验 SHA256：`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。
保存时复制为 checkpoint 内的 `openpi_tokenizer.model`，迁移机器后优先读取这个文件。
文本输入不再依赖 HF tokenizer 的 regex 行为。

本轮实验从 PI05-base 重新训练，不支持此前实现生成的旧 LAP 模型，也不进行旧 embedding
权重换算。当前实现保存的 LAP checkpoint 可用于阶段二初始化或正常断点续训。
共享权重在扩词表、参数更新和保存重载后均保持绑定。若 checkpoint 中的 embedding 与
LM head 数值不一致，加载会报错；此前未绑定版本的第一阶段需从 PI05-base 重新训练。

## 配置

阶段、LAM 与 tokenizer 的配置字段：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `policy.training_stage` | `post-training` | `pre-training` 或 `post-training` |
| `policy.lam_model_path` | `null` | UniVLA stage-2 LAM 的本地 `.ckpt` 文件，仅预训练需要 |
| `policy.lam_camera_key` | `null` | 单个 RGB 相机全名或短名，省略时自动选主视角 |
| `policy.tokenizer_path` | `google/paligemma-3b-pt-224` | latent token ID 注册用的 PaliGemma HF tokenizer 目录或 Hub ID |
| `policy.tokenizer_model_path` | `null` | 官方文本 SentencePiece `.model` 文件，省略时自动缓存 |
| `policy.discrete_state_input` | `false` | 是否把归一化状态写入 prompt，LIBERO 官方设为 false |
| `datasets.<name>.image_augmentation` | `true` | 数据集 JSON 中逐来源控制训练图像增强；验证和推理关闭 |
| `datasets.<name>.use_relative_actions` | `false` | 数据集 JSON 中逐来源控制是否额外计算 `action - state` |
| `policy.gradient_accumulation` | `1` | 累计多少个每卡 batch 后更新一次，预训练和后训练均支持 |
| `policy.attention_backend` | `eager` | 训练注意力：`eager`、`sdpa`、`flash_attention_2` |

### A100 单卡显存优化

已有 BF16 autocast 的训练可增加：

```bash
--policy.attention_backend=flash_attention_2 \
--policy.gradient_checkpointing=true
```

保持 `accelerate launch --mixed_precision=bf16`。无需安装第三方 `flash-attn`：
这里使用 PyTorch 自带的 FlashAttention。PyTorch 2.10+ 将可变长度块打包成一次
Flash 调用；更早的受支持版本使用逐块 SDPA Flash 内核，不会静默回退为 dense attention。
不支持的设备或精度会明确报错。非 CUDA / FP32 训练可选择 `sdpa` 或保留 `eager`。

保持 `batch_size=8`、`policy.gradient_accumulation=4`，有效 batch 仍为 32。
如果还需降低激活显存，可以同时改为 4 和 8，有效 batch 与更新步数口径仍不变。
梯度检查点通过重计算降低激活显存，会增加计算量；FlashAttention 不消除模型权重、
梯度和 Adam 优化器状态的显存占用。

Flash 路径按原始 block mask 打包：观测 prefix 双向、动作块能看见 prefix 与整个动作块、
PI0 状态 token 独立成块，LAP latent suffix 逐 token 因果。padding token 不作为 key，
其 query 输出置零且不参与监督；真实 token 的注意力关系不变。不会额外降低精度，
原有 FP32 权重、FP32 norm/视觉模块和 BF16 autocast 设置继续生效。
默认 `eager` 和动作推理/KV cache 路径保持不变。切换注意力后端不需要转换权重。

融合内核的浮点累加顺序与 eager 不同，不能保证逐位相同或仅靠单元测试保证最终任务成功率。
本仓库提供前向、loss、梯度、因果泄漏和重计算一致性测试，以及真正调用 CUDA Flash
的 BF16 回归与显存测试。服务器上先运行：

```bash
uv run --no-sync pytest tests/policies/pi0_pi05/test_pi_attention.py tests/policies/pi05_lap/test_pi05_lap.py -q
```

CPU 测试会跳过真实 CUDA 用例；CPU 打包测试只使用数学参考内核，不代表 A100 实测。
对真实 checkpoint 的长期精度仍应对比相同数据、种子与有效 batch 下的验证指标。

自动选择相机时依次优先 `image`、`base_0_rgb`、`primary`、`front`、
`agentview_image`、`image_primary`，否则使用 metadata 中的第一个 RGB 相机。
不接受相机列表；显式填写的相机不存在或不是 RGB 时直接报错。

本地 tokenizer 传目录，例如 `--policy.tokenizer_path=/mnt/models/paligemma-tokenizer`，
目录内应有 `tokenizer.json`、`tokenizer_config.json` 等 tokenizer 文件。
必须使用与 PI05 权重匹配的 PaliGemma tokenizer；该路径只用于 latent 词表初始化。
文本 processor 始终使用上面的官方 SentencePiece 模型。
从 LAP checkpoint 继续训练或做动作微调时，自动使用其中保存的 `latent_tokenizer/` 和 ID 映射，
无需再提供原始 tokenizer 路径。

## 预训练

从原 PI05-base checkpoint 初始化，也可从当前实现生成的 PI05 LAP checkpoint 继续。
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
从 PI05-base 初始化时处理 `vision_tower.vision_model.*` 到当前 `vision_tower.*` 的名称变化，
仅映射当前模型中确实存在的目标 key，并继续严格检查缺失项、额外项和 tensor shape。
原始 PI05-base 的 `config.json` 保持 `type: pi05`，其 processor 也无需手工改写；
加载器负责配置转换，将原生状态拼接和 HF 文本 tokenizer 步骤替换为官方输入流程。

共享 `pi_gemma.py` 对策略已构造的四维注意力 mask 直接使用，保留 prefix 和 latent suffix 的
注意力关系；需要自动构造 causal mask 时，按 Transformers 实际接口决定是否传入
`cache_position`，兼容不再接收此参数的版本。

## 动作微调

将启动脚本中的 `policy.path` 改为预训练输出的 `pretrained_model` 目录，
`policy.training_stage` 改为 `post-training`，并更换 `output_dir`。
这一阶段只使用真实动作的原生 PI05 flow-matching 损失，不加载 LAM，也不采样未来图像。
动作专家从第一阶段保留的 PI05 权重开始训练，VLM 从预训练结果继续；
`train_expert_only`、`freeze_vision_encoder` 等现有选项仍然有效。

两个阶段都可以使用 `--policy.gradient_accumulation=4`。例如 `batch_size=8` 时，
单卡有效 batch 为 32；保存、评估、学习率和训练总步数均按优化器更新次数计算。
续训会恢复该配置，也可以显式覆盖；数据位置按已消费的 batch 数恢复。

动作推理使用原有 `select_action` / `predict_action_chunk`。
预训练模型可用 `predict_latent_actions(processed_batch)` 从当前观测自回归生成四个码本索引；
它不需要未来图像或 LAM 权重。动作推理需切到 `post-training`。

后训练可以设置 `--eval_steps=1000`，并在数据 JSON 中设置 `"eval_split": 0.05`，
每隔 1000 步对留出 episode 中所有完整 horizon 执行动作推理，记录动作 RMSE、
夹爪准确率和累计增量误差（单数据集只记总指标，多数据集额外记各来源指标）。夹爪暂按 `{-1, +1}`、阈值 0；
7/14 维动作分别按单臂/双臂计算，排除补齐维度和不足 H 的末尾。
评估使用真实的完整推理与反归一化流程，不设固定 10 组上限。
详见 [后训练评估说明](../../../../docs/dataset_config.md#后训练中的完整动作推理评估)。

## 启动

安装依赖：`uv sync --locked --extra training --extra pi05-lap`。
仓库根目录的 `train_pi05_lap.sh` 从原生 PI05-base 开始 baseline，
`train_pi05_lap_s2.sh` 从当前实现重新生成的 latent 预训练结果开始阶段二，两者均为新训练。
阶段二脚本的 `policy.path` 默认指向 `PI05-lap-pretrain-openpi/checkpoints/last/pretrained_model`，
需与本轮阶段一实际保存路径一致。
输出目录和实验名以 `-openpi` 区分已有实验，不覆盖旧训练。
两份脚本使用同一数据配置、batch=4、梯度累积=8（单卡有效 batch=32）、
FP32 模型参数配合 BF16 autocast、FlashAttention、视觉编码器可训练、30000 次优化器更新，
峰值学习率 `2.5e-5`，warmup=1000，cosine decay 到第 30000 步的 `2.5e-6`。
这些训练超参数为本项目对照实验设置，不声称与官方完整训练配方相同。
按服务器实际位置修改模型、tokenizer 和数据路径后运行：

```bash
bash train_pi05_lap.sh --policy.chunk_size=12 --policy.n_action_steps=12
# 阶段二使用当前实现的预训练权重，并按需要指定动作步数：
bash train_pi05_lap_s2.sh --policy.chunk_size=12 --policy.n_action_steps=12
```

启动脚本选中的数据 JSON 使用 `"video_backend": "pyav"`。如果 TorchCodec 报
`Could not load libtorchcodec`，并提示缺少 `libavutil.so.*`，可使用这一现有后端
读取视频；该设置只影响数据解码，PI05 和 UniVLA 仍在 CUDA 上运行。

服务器 CUDA 训练和实际 checkpoint 的效果需在目标服务器验证。

## SwanLab 日志

训练依赖已包含 SwanLab。在服务器当前训练环境中安装并登录：

```bash
pip install -e '.[training,pi05-lap]'
swanlab login
bash train_pi05_lap.sh
```

使用 uv 管理环境时可运行 `uv sync --extra training --extra pi05-lap`，
然后用 `uv run swanlab login` 和 `uv run bash train_pi05_lap.sh`。

新训练或其他策略使用同一组参数即可：

```bash
--swanlab.enable=true \
--swanlab.project=pi05-lap \
--swanlab.mode=online \
--wandb.enable=false
```

实验名沿用 `job_name`，每 `log_freq` 步记录训练 loss、latent token accuracy、
梯度范数、学习率、吞吐和显存等现有指标，也记录评估指标。
仅主进程写日志。启动时打印实验链接；本地日志写入 `output_dir/swanlog/`。
`swanlab.workspace` 可选填组织空间名，省略时使用当前账号的个人空间。

从旧 W&B checkpoint 切换时无需修改旧 JSON，优先启用 SwanLab 并关闭 W&B。
模型、优化器和全局 step 按原流程恢复；SwanLab 创建新实验，从恢复的 step 开始记录，
不会自动迁移旧 W&B 曲线。新的 `train_config.json` 会保存 `swanlab.run_id`，
以后从这些 checkpoint 续训会复用相同实验；也可用 `--swanlab.run_id=<ID>` 显式指定。

无外网时使用 `--swanlab.mode=offline`，无需登录；训练后可以用
`swanlab sync <output_dir>/swanlog/<run目录>` 上传本地实验。
SwanLab 公有云不支持 checkpoint 文件上传，完整 checkpoint 继续保存在本地。
环境评估产生的 MP4 会转换为 GIF 上传到 SwanLab，原 MP4 保留。

训练入口会在正常结束时提交 `success`，在 Ctrl+C / SIGTERM 时提交 `aborted`，
在初始化或训练报错时提交 `crashed`；只有主进程调用 SDK，提交后仍保留原异常及退出状态。
按一次 Ctrl+C 后，请等待终端中 `Closing SwanLab experiment ...` 的同步收尾。
再次中断同步、网络断开或 `kill -9` 都可能导致云端暂时继续显示“运行中”。
对于已经停止的旧训练，可在 SwanLab 实验名旁点击终止按钮，将状态改为“中断”；
这只改变云端记录，不会终止服务器进程。若不手动处理，官方规则是约 30 分钟无任何日志
（包括系统指标）上传后自动标记中断，见 [SwanLab FAQ](https://docs.swanlab.cn/guide_cloud/experiment_track/FAQ.html)。
