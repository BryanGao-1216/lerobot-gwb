# SmolVLA LAP

Python 包及 CLI 策略类型为 `smolvla_lap`。原来的 `smolvla` 策略不变。

## 两个阶段

- `pre-training`：从原始 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 初始化 VLM。输入与 SmolVLA 一致：当前所有相机图像、任务文本、当前机器人 state。冻结的 UniVLA stage-2 LAM 从指定单相机的 `(t, min(t+H, episode末帧))` 提取 4 个任务相关 token（每个 ID 为 0–15）。VLM 用 teacher forcing 自回归预测这 4 个 token，只计算 token 交叉熵。动作专家冻结；VLM、输出头、state projection 训练，视觉编码器由 `freeze_vision_encoder` 控制。
- `post-training`：加载 LAP 预训练 checkpoint，保留 VLM 和 state projection，使用原来的 SmolVLA flow matching 训练连续动作专家。训练标签只有真实机器人动作；不运行 LAM，不预测或输入 latent token。

固定 H 的单位是数据集帧数，默认 `12`，与动作专家的 `chunk_size` 无关。未来帧只提供给 LAM。LAM 输入采用原始 RGB 帧经 PIL bilinear 缩放至 224×224，范围 `[0,1]`，内部执行 ImageNet 归一化；伪标签不受 VLM 随机图像增强影响。末尾不足 H 时使用同 episode 最后一帧，包含末帧到自身的帧对。

后训练针对 7D 相对位姿 + 夹爪动作：episode 末尾越界的动作位置在归一化前替换成 `[0,0,0,0,0,0,last_gripper]`，作为有效静止命令参与监督。原有 episode 内动作不做修改；不自动把绝对动作转成相对动作。

## 安装及启动

在服务器上安装本仓库及其可选依赖：

```bash
uv sync --locked --extra smolvla-lap --extra training
```

UniVLA 和 DINOv2 必要推理源码已包含在 `lam/`，无需另行安装上游仓库、Lightning、T5 或 FlashAttention。只需提供 UniVLA **stage-2 LAM** 权重，例如 `lam-stage-2.ckpt`；stage-1 或完整 VLA policy 权重会明确报错。所有 LAM 参数严格加载，教师始终在当前进程的 CUDA 设备上用 FP32 推理，并保持冻结。`lam_batch_size` 控制教师推理的分块大小，不改变 VLM 训练 batch size。

启动脚本为仓库根目录的 `train_smolvla_lap_lr.sh`，与 `train_pi.sh` 一样，直接在 `accelerate launch` 命令中修改参数：

```bash
bash train_smolvla_lap_lr.sh
```

脚本默认预训练；先修改 `--policy.lam_model_path`、`--policy.lam_camera_key`、数据集路径及输出目录。相机参数只传一个名字，也接受 `observation.images.front` 这样的完整字段。

切换后训练时，将 `--policy.type=smolvla_lap` 替换为 `--policy.path=/path/to/lap_pre/checkpoints/last/pretrained_model`，将 `--policy.training_stage` 改为 `post-training`，并修改 `--output_dir`。这个 checkpoint 必须由 LAP 策略生成；后训练不使用 LAM 相关参数。

原始 SmolVLM 的本地目录直接填写在 `--policy.vlm_model_name`。加载 checkpoint 时也可删除此参数，沿用 checkpoint 保存的 VLM 名称。多 GPU 时修改 `CUDA_VISIBLE_DEVICES` 和 `--num_processes`，并添加 `--multi_gpu`；`--batch_size` 是每张 GPU 的 batch size。默认使用 BF16 训练 VLM；LAM 不受外层 autocast 影响。

主要参数：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--policy.training_stage` | 训练阶段 | 脚本为 `pre-training`，策略配置为 `post-training` |
| `--policy.lam_type` | LAM 类型 | `univla` |
| `--policy.lam_model_path` | UniVLA stage-2 权重路径 | 预训练必填 |
| `--policy.lam_camera_key` | 单个 RGB 相机字段 | 脚本为 `front`，策略配置中必填 |
| `--policy.lam_frame_gap` | 两帧间隔 | `12` |
| `--policy.lam_batch_size` | 教师推理分块大小 | `8` |
| `--policy.path` | 已有 LAP checkpoint | 后训练时填写预训练 checkpoint |
| `--policy.vlm_model_name` | 原始 VLM 名称或本地目录 | 原始 SmolVLM2-500M-Video-Instruct |

保持默认 SmolVLA 的 16 层 VLM 设置；如需使用全部层，可在首次预训练时在启动命令中增加 `--policy.num_vlm_layers=-1`。后训练和恢复训练应保持 checkpoint 的层数与模型结构。`--policy.train_expert_only` 和 `--policy.train_state_proj` 只控制后训练阶段；预训练会显式训练 VLM 和 state projection。

恢复同一阶段的训练（恢复优化器、调度器和步数）应使用 LeRobot 的 `--resume=true --config_path=/path/to/train_config.json`；从预训练切换后训练使用 `--policy.path`，开始新的优化器及调度器。

预训练日志包含 `loss`、`latent_token_accuracy`、`latent_sequence_accuracy`，后两项为 teacher-forced 准确率。`policy.generate_latent_actions(preprocessed_batch)` 可以仅凭当前输入贪心生成 4 个 LAM code ID，用于真正自回归检查，不依赖 LAM 权重或未来帧。

## TensorBoard

启动脚本已默认开启 `--policy.tensorboard_enable=true`，每 10 步写入一次 scalar。两个训练阶段都写到各自的 `<output_dir>/tensorboard/`；多 GPU 仅由主进程写事件文件，恢复训练沿用现有日志并清理恢复步之后的旧记录。

可查看的曲线包括 `train/loss`、`train/lr`、`train/grad_norm` 和 `train/gpu_mem_gb`。预训练还会记录 `train/latent_token_accuracy` 和 `train/latent_sequence_accuracy`，它们是 teacher-forced 准确率。开启验证集评估时会记录 `eval/loss`。

在训练所用环境中，另开终端启动仓库根目录的查看脚本：

```bash
bash tensorboard.sh
```

`tensorboard.sh` 的 `--logdir` 已与当前训练脚本的输出目录对应。修改训练的 `--output_dir` 后同步修改这里；也可指向多个训练输出目录的共同父目录以比较实验。浏览器访问 `http://127.0.0.1:6006`。训练在远程服务器时，先在本机建立端口转发：

```bash
ssh -N -L 6006:127.0.0.1:6006 user@server
```

`--policy.tensorboard_log_dir` 可以指定其他日志目录，相对路径以 `output_dir` 为起点。写入频率通过 `--policy.tensorboard_log_freq` 修改，刷新磁盘间隔默认为 30 秒。参数和梯度直方图默认关闭；需要时添加 `--policy.tensorboard_log_parameters=true`、`--policy.tensorboard_log_gradients=true`，通过 `--policy.tensorboard_histogram_freq` 控制频率（默认 1000 步）。

两阶段的 checkpoint 都保留完整 LAP 策略结构和 special-token 映射。冻结教师不写入 checkpoint；恢复预训练时仍需提供有效的 `lam_model_path`。后训练和连续动作推理不需要 LAM 权重。当前接入支持非 streaming 的 LeRobot 数据集，不包含 RLDS。

策略训练部分未执行 CUDA 训练或训练测试；需要在服务器验证实际显存、吞吐和收敛表现。
