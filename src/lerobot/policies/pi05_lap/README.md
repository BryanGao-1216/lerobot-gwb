# PI05 LAP

多数据集混合训练可使用 `--dataset.config_path=dataset/libero.json`，各数据集独立配置字段映射、相机和抽样权重。详见 [数据配置说明](../../../../docs/dataset_config.md)。

JSON 中每个数据集的 `camera_fields` 把 VLM 视角映射到数据集原始相机字段，可以选择部分相机，并用 `null`、`""` 或省略槽位表示空视角。`lam_camera_key` 填写其中一个非空原始相机字段，用于 latent 预训练；不能选择 VLM 输入之外的相机。只读取实际选中的相机。

从当前 `pi05/` 复制并扩展的两阶段策略，原 PI05 策略保持独立。
输入仍然是 PI05 的当前图像、任务指令和离散化状态，使用原有 processor 和归一化流程。

## 配置

阶段、LAM 与 tokenizer 的配置字段：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `policy.training_stage` | `post-training` | `pre-training` 或 `post-training` |
| `policy.lam_model_path` | `null` | UniVLA stage-2 LAM 的本地 `.ckpt` 文件，仅预训练需要 |
| `policy.lam_camera_key` | `null` | 单个 RGB 相机全名或短名，省略时自动选主视角 |
| `policy.tokenizer_path` | `google/paligemma-3b-pt-224` | 初始化时使用的 PaliGemma tokenizer 目录或 Hub ID |
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
默认 `eager` 和动作推理/KV cache 路径保持不变，旧 checkpoint 不需要转换。

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
必须使用与 PI05 权重匹配的 PaliGemma tokenizer；该路径同时用于文本 processor 和
latent 词表初始化，并覆盖原 PI05 processor 配置中保存的 Hub ID。
从 LAP checkpoint 继续训练或做动作微调时，自动使用其中保存的 `latent_tokenizer/` 和 ID 映射，
无需再提供原始 tokenizer 路径。

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
加载器兼容旧 checkpoint 中 `vision_tower.vision_model.*` 到当前 `vision_tower.*` 的名称变化，
仅映射当前模型中确实存在的目标 key，并继续严格检查缺失项、额外项和 tensor shape。
原始 checkpoint 的 `config.json` 保持 `type: pi05`，processor 中的 Google tokenizer 名称也无需手工改写；
启动参数 `policy.type=pi05_lap`、`policy.tokenizer_path=...` 会负责配置转换和 tokenizer 覆盖。

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
夹爪准确率和累计增量误差（总指标和各数据集指标）。夹爪暂按 `{-1, +1}`、阈值 0；
7/14 维动作分别按单臂/双臂计算，排除补齐维度和不足 H 的末尾。
评估使用真实的完整推理与反归一化流程，不设固定 10 组上限。
详见 [后训练评估说明](../../../../docs/dataset_config.md#后训练中的完整动作推理评估)。

## 启动

安装依赖：`uv sync --extra training --extra pi05-lap`。
仓库根目录的 `train_pi05_lap.sh` 为单命令续训模板：从 `checkpoints/last` 恢复，
保留 checkpoint 中的预训练、LAM、相机、H 和优化器配置，训练到累计 50 万步。
修改其中 `config_path` 后运行：

```bash
sh train_pi05_lap.sh
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
sh train_pi05_lap.sh
```

使用 uv 管理环境时可运行 `uv sync --extra training --extra pi05-lap`，
然后用 `uv run swanlab login` 和 `uv run sh train_pi05_lap.sh`。

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
