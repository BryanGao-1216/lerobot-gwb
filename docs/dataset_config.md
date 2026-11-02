# JSON 数据集配置与混合训练

当前入口支持本地/服务器启动的 PI05 和 PI05 LAP，包括 LAP 的预训练与动作后训练。数据仍然是 LeRobot 格式，不需要合并或复制原始数据。

在项目根目录执行训练，使用一个参数选择数据配置：

```bash
--dataset.config_path=dataset/libero.json
```

`dataset/` 只存放 JSON 配置。已提供的 `libero.json` 包含 LIBERO、ALOHA boil 和 ALOHA solid，使用现有服务器路径与相机字段；可以复制为其他名字，在启动脚本中选择。数据目录要指向包含 `meta/`、`data/`、`videos/` 的 LeRobot 根目录。JSON 内的相对路径相对于该 JSON 所在目录解析。

使用 JSON 时，数据参数以文件内容为准，不要再通过启动脚本传 `dataset.repo_id/root/video_backend` 等覆盖项。旧的单数据集 CLI 仍然可以使用。

## 当前 LIBERO 与 ALOHA 配置

`dataset/libero.json` 中三个来源的权重均为 `1.0`，抽样概率各约为 1/3。ALOHA boil 根目录为 `/gaowenbin/datasets/solid/boil/boil_0613`，ALOHA solid 为 `/gaowenbin/datasets/solid/solid/measure_solid_0703_3`。`repo_id=boil` / `solid` 是读取本地数据所用的标识，没有指定 Hub 仓库。

| 统一 VLM 视角 | LIBERO 原始字段 | ALOHA 原始字段 |
| --- | --- | --- |
| `observation.images.image` | `observation.images.image`（头部） | `observation.images.cam_high` |
| `observation.images.image2` | `observation.images.image2`（右手） | `observation.images.cam_right_wrist` |
| `observation.images.image3` | 空，自动掩码 | `observation.images.cam_left_wrist` |

ALOHA 使用 `cam_high` 生成 LAM 标签。动作和状态均从原始“左臂 7 维、右臂 7 维”重排为“右臂 7 维、左臂 7 维”，不读取 velocity/effort。关节字段仍保持原始表示；该重排不会转换成末端相对位姿。latent 预训练的监督来自图像 LAM 标签，动作后训练前需要另行确认、统一动作表示。

提供的 ALOHA metadata 为 `v2.1`，当前读取器要求 `v3.0`。在服务器项目根目录执行格式转换后再启动训练，配置中的根目录无需改变：

```bash
uv run python -m lerobot.scripts.convert_dataset_v21_to_v30 \
  --repo-id=boil \
  --root=/gaowenbin/datasets/solid/boil/boil_0613 \
  --push-to-hub=false
```

转换脚本会把旧数据移到同级的 `boil_0613_old`，转换后的数据放回原路径。转换后用于 PI05 默认归一化的 action/state 统计量仍需包含 `q01/q99`；格式转换不会凭空补出缺失的分位数。

## 添加数据集

在 `datasets` 下新增一个命名配置块。例如下面两份数据的动作来自不同字段，相机数量也不同：

```json
{
  "video_backend": "pyav",
  "return_uint8": true,
  "use_imagenet_stats": true,
  "image_transforms": {"enable": false},
  "eval_split": 0.0,
  "tolerance_s": 0.0001,
  "datasets": {
    "libero": {
      "repo_id": "libero_only",
      "root": "/gaowenbin/datasets/libero",
      "weight": 1.0,
      "action_fields": [{"key": "action"}],
      "state_fields": [{"key": "observation.state"}],
      "camera_fields": {
        "observation.images.image": "observation.images.image",
        "observation.images.image2": "observation.images.image2"
      },
      "lam_camera_key": "observation.images.image"
    },
    "another_dataset": {
      "repo_id": "your_account/another_dataset",
      "root": "/path/to/another_lerobot_dataset",
      "weight": 3.0,
      "action_fields": [
        {"key": "action.delta_pose", "indices": [0, 1, 2, 3, 4, 5]},
        {"key": "action.gripper"}
      ],
      "state_fields": [{"key": "observation.robot_state"}],
      "camera_fields": {
        "observation.images.image": "observation.images.front",
        "observation.images.image2": null
      },
      "lam_camera_key": "observation.images.front"
    }
  }
}
```

第二块是结构示例，路径和源字段需要换成真实数据。每个块可再设置 `episodes`（episode 编号列表）、`revision`、`video_backend`、`tolerance_s`、`eval_split`。后三项覆盖 JSON 顶层的默认值。`repo_id` 即使读取本地数据也需要填写。

| 字段 | 含义 |
| --- | --- |
| `weight` | 正数，相对抽样权重。上例两个数据集约占 25% 和 75%，不再乘以数据集帧数。 |
| `action_fields` | 按列表顺序读取并拼接为 `action`；`indices` 可选，从 0 开始，用于选取或重排维度。 |
| `state_fields` | 同样按顺序拼接为 `observation.state`。 |
| `camera_fields` | **左边是统一 VLM 视角槽位，右边填数据集内的原始相机字段**；左侧必须以 `observation.images.` 开头。右侧允许 `null` 或 `""` 表示空视角，也可以省略不用的槽位。 |
| `lam_camera_key` | 单个 LAM 监督相机的原始字段，必须在 `camera_fields` 的非空右侧值中。省略或填 `null` 时，使用第一个非空视角。只影响 LAP 预训练。 |

本次没有添加采样步长参数。动作 chunk 使用连续帧；LAM 使用 `t` 和 `t + policy.chunk_size`，尾部截断到当前 episode 最后一帧，动作尾部仍沿用 PI05 的原有处理方式。

## 每个数据集的相机配置

`camera_fields` 决定 VLM 接收哪些**当前帧**。每个数据集可以配置一台、两台、三台或更多相机，无需另设数量参数，也不必填写数据集中所有相机。下面是三个数据集配置块中的相机部分，其他字段按上面的完整示例填写：

```json
{
  "one_camera_dataset": {
    "camera_fields": {
      "observation.images.image": "observation.images.front",
      "observation.images.image2": null,
      "observation.images.image3": ""
    },
    "lam_camera_key": "observation.images.front"
  },
  "two_camera_dataset": {
    "camera_fields": {
      "observation.images.image": "observation.images.front",
      "observation.images.image2": "observation.images.wrist"
    },
    "lam_camera_key": "observation.images.wrist"
  },
  "three_camera_dataset": {
    "camera_fields": {
      "observation.images.image": "observation.images.front",
      "observation.images.image2": "observation.images.left_wrist",
      "observation.images.image3": "observation.images.right_wrist"
    },
    "lam_camera_key": "observation.images.front"
  }
}
```

这里三个数据集的 VLM 相机槽位合计为三个：使用左侧统一字段的并集，缺失的槽位补空图并设置注意力掩码。同一种视角请使用一致的左侧名称，例如都把主视角映射为 `observation.images.image`。

每个数据集的 `lam_camera_key` 只能从已选的非空 VLM 输入视角中指定一个 RGB 相机，填写对应的数据集原始字段。即使数据集中还有其他相机，只要没有选入 `camera_fields`，也不能用于 LAM；配置不符时会在加载模型前报错。为兼容旧配置，仍接受指向非空视角的左侧别名，但原始字段匹配优先。使用 JSON 混合配置时，各数据集的这个字段取代全局 `policy.lam_camera_key` 的选择。

- LAP 预训练：每个数据集至少需要一个非空 VLM 视角。只加载 `camera_fields` 选中的原始相机；其中 LAM 相机查询 `t`、`t+H` 两帧，VLM 只接收它的 `t` 帧。
- 动作后训练：只加载 `camera_fields` 中非空的原始相机，均读取当前帧。允许所有槽位为空或使用 `{}`，此时该数据集不提供图像输入，`lam_camera_key` 应省略或设为 `null`。
- 空槽位不读取相机文件，返回的占位图带有无效视角掩码；未写入该数据集配置、但由其他数据集使用的槽位同样处理。
- 未选择的图片列在 Parquet 读取时排除，未选择的视频不解码，也不要求其视频文件存在或下载。原始数据集的 metadata 不会被改写。

## 不同维度与归一化

动作按数据中保存的原始值读取，当前混合入口的 `policy.use_relative_actions` 需要保持 `false`。字段映射只负责选维、重排和拼接，不转换动作表示、坐标系、单位或旋转表示。动作后训练时需先保证各来源相同输出位置具有一致的控制含义；关节动作与 LIBERO 末端相对动作无法仅靠重排完成统一。

各数据集单独使用自己的统计量归一化，PI05 默认使用 `q01/q99`。每个被选择的向量字段都要有对应统计量；缺失时会在模型加载前报错，不会套用另一个数据集的统计量。

向量补零到本次配置中最大的动作/状态维度，补齐位置在归一化后仍为零；动作损失只计算该样本的有效维度。超出策略的 `max_action_dim/max_state_dim` 会直接报错，不会截断标签。不同分辨率的相机先等比缩放、补边到策略分辨率。相机集合取统一字段的并集，缺少的视角用空图和注意力掩码处理。教师帧对在学生图像增强前提取。

先按权重抽数据集，再在其可用帧中均匀抽样，采用有放回采样。一轮以所有数据集可用帧数之和为基数，向上凑齐到全局 batch 的整数倍。多进程使用同一确定性采样序列，由 Accelerate 切分 batch。

设置 `eval_split` 后，每个数据集分别按 task 留出完整 episode，训练和验证不会拆分同一个 episode。验证遍历留出的数据，不按训练权重重复抽样。统计量沿用原数据集提供的统计量，不额外重算训练子集统计量。

## 梯度累积

在训练脚本中配置 `--policy.gradient_accumulation=N`，默认 `1`，必须是正整数。PI05 LAP 的预训练和后训练共用该参数。例如：

```bash
--batch_size=8
--policy.gradient_accumulation=4
```

每张 GPU 每次加载 8 条样本，连续处理 4 个 micro-batch 后才更新一次参数；单卡名义有效 batch 为 32，双卡为 64。一般公式为 `batch_size × GPU/进程数 × gradient_accumulation`。每个 micro-batch 的计算图在反向传播后释放，不会将 N 个 batch 的图像和计算图一起保留。跨越数据轮次边界时仍累计满 N 个 batch，较小的末批按实际样本数参与平均。

`steps`、学习率 warmup/decay、`save_freq`、`eval_steps`、`env_eval_freq` 和 `log_freq` 都按优化器实际更新次数计算。梯度同步、裁剪、优化器更新和学习率调度只在完整累计窗口结束时进行。日志中的 loss 是累计窗口的平均原始 loss，不会额外除以 N。AMP 溢出跳过权重更新时，不推进上述步数或学习率。

checkpoint 保存累计步数配置及已消费的 micro-batch 数，仅在完整更新后保存。续训不传此参数会沿用 checkpoint 中的值；旧 checkpoint 默认按 `1` 处理。在数据配置、每卡 micro-batch 大小和进程数不变时，可以通过该参数调整 N，数据读取位置仍按已经消费的 batch 恢复，不会误按新的 N 放大历史步数。学习率沿用原配置，不自动按有效 batch 调整。

## 后训练中的完整动作推理评估

PI05 和 PI05 LAP 的动作后训练使用现有 `eval_steps` 控制评估间隔，`0` 关闭。PI05 LAP 的 latent 预训练不运行以下动作指标，仍保留原有的验证 loss 入口。

在训练脚本的后训练参数中添加，例如：

```bash
--policy.training_stage=post-training
--dataset.config_path=dataset/libero.json
--eval_steps=1000
```

在选中的数据 JSON 顶层设置 `"eval_split": 0.05`，按每个来源、每个 task 留出约 5% 的完整 episode（数量向上取整）；也可以在某个来源块中用 `eval_split` 单独覆盖。当前仓库 JSON 的默认值仍是 `0.0`，启用评估时需要修改。小 task 至少会留出一个 episode，需保证训练集合仍有数据。

每次评估遍历留出的全部合格起始帧，H 为 `policy.chunk_size`。一个长 L 的 episode 有 `max(0, L-H+1)` 个候选 chunk，相邻 chunk 可以重叠。末尾不足 H 的帧不参与评估；刚好剩余 H 帧时参与。不存在固定 10 组或其他固定数量上限，`max_eval_samples` 仅用于旧验证 loss 路径，不限制动作评估。

评估使用无图像增强的验证观测、任务和状态，调用当前模型的 `predict_action_chunk`，按 `policy.num_inference_steps` 完成全部 flow-matching 迭代。真实 action 不作为模型输入。预测动作经过原有后处理，按每条样本的来源反归一化后，与原始标签比较。固定每个样本的初始推理噪声，并在评估结束后恢复训练模式与随机数状态。多进程下去掉末批为对齐而重复的样本，每条验证 chunk 只计一次。

每臂约定为 6 维增量动作和 1 维夹爪，支持有效动作维度 7 或 14。混合 batch 中 LIBERO 补齐到 14 维的后半部分不计算指标。按当前约定，所有来源（包括转换后的 ALOHA）都使用增量动作，夹爪标签为 `{-1, +1}`。

| 指标名称 | 计算方式 |
| --- | --- |
| `action_rmse` | 每臂对 H×6 个误差计算 RMSE。14 维动作分别使用 `0..5` 和 `7..12`，再取两臂平均。 |
| `gripper_accuracy` | 预测和标签均以 `>0` 为一类、`<=0` 为另一类；7 维用索引 6，14 维用 6、13，按 H 和夹爪数量平均。值域为 0～1。 |
| `cumulative_action_error` | 每个有效增量维度沿 H 求和，计算预测累计值与标签累计值之差的绝对值，再平均六维和有效臂。 |

先计算每条 chunk 的三个指标，再按 chunk 数量平均。SwanLab 和终端同时记录总指标与分来源指标，例如 `eval/action_rmse`、`eval/libero/action_rmse`、`eval/aloha_boil/gripper_accuracy`，以及对应的 `num_action_chunks`。总体是所有验证 chunk 的平均，不使用训练抽样权重。

## 续训和部署

checkpoint 的 `train_config.json` 保存解析后的数据配置，包括数据集顺序和权重；processor 文件保存各数据集自己的归一化统计量。

- 续训不传 `dataset.config_path`：直接使用 checkpoint 快照，即使原 JSON 已被修改或删除也不会自动替换。
- 显式传入 `dataset.config_path`：重新读取文件。若配置发生变化，保留模型、优化器、学习率调度器和累计训练步数，从新采样序列第 0 轮开始，并记录新的采样起点。
- 配置、每卡 micro-batch size 和进程数相同：按 checkpoint 记录的已消费 batch 数恢复数据抽样位置，兼容梯度累积。旧 checkpoint 按每个训练步读取一个 batch 恢复。

数据源按 JSON 的顺序编号为 `dataset_index=0,1,...`。独立部署混合数据训练出的模型时，必须选择目标数据集的统计量，并按训练配置映射输入字段、补齐维度。例如加载 processor 时：

```python
preprocessor_overrides = {"normalizer_processor": {"dataset_index": 0}}
postprocessor_overrides = {"unnormalizer_processor": {"dataset_index": 0}}
```

训练 batch 会自动携带每个样本的 `dataset_index`，无需手动指定。单来源配置可以自动选择其统计量。当前混合入口提供离线验证；在线环境评估需要单独选择部署数据源，训练时设置 `env_eval_freq=0`。Streaming、HF Jobs、RA-BC 帧权重和其他策略尚未接入此混合入口。
