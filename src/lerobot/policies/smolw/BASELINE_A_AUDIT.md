# SmolW-A / SmolVLA 基线审计（2026-09-10）

结论：**同一初始权重、有效配置、数据和运行环境下，A 的动作学习与推理路径可以代表
本仓库 SmolVLA 基线。它不等于旧 SmolVLA 启动脚本的完整实验，也不能据此保证官方公布的成功率。**

比较对象是本仓库 `policies/smolvla` 和当前训练、评估框架；没有把它们认证为某个上游发布版本。
本地没有服务器上的完整 SmolVLA / SmolW base、LIBERO 数据或 CUDA 双卡环境，因此服务器
artifact 是否一致、完整训练和 LIBERO 成功率仍需在实际环境验证。

## 实现逐项核对

| 环节 | A 与同配置 SmolVLA 的关系 |
| --- | --- |
| 模型构造 | A 直接调用 `SmolVLAPolicy.__init__`，模型类型就是 `VLAFlowMatching` |
| 网络参数 | 同样的 VLM、vision encoder、connector、expert、state/action/time heads；无 motion/z 参数 |
| 初始化随机数 | 同配置构造时无额外 motion 初始化；小模型测试验证参数和 RNG 状态完全相同 |
| 冻结与 train/eval | 使用原 `set_requires_grad` 和 `train`，服从相同冻结开关 |
| 数据时间戳 | observation `[0]`，action `[0, …, chunk_size-1]`，reward `None` |
| 数据采样 | 当前训练器对两者均使用同一个 EpisodeAwareSampler；A 的 `drop_n_last_frames=0` 对齐原版缺省值 |
| 图像 | 直接调用原 `prepare_images`；相机顺序、当前帧、缺失相机、有效 mask、resize/padding 和 [-1,1] 变换相同 |
| 状态、动作和语言 | 原预处理；相同补维、归一化映射、task 换行、tokenizer 配置与 language padding |
| episode 尾部 | A 加载旧 base 时移除 stationary action padding；保留原始动作和 `action_is_pad` |
| attention | 原 prefix/suffix、mask、RoPE、位置索引、self/cross attention，无额外 token |
| flow matching | 原 action noise、time 分布、插值、velocity target 和 loss；无 z noise 或 z loss |
| loss reduction | 原真实动作维度截取、有效时间步分母、mean/none；A 只额外记录 `action_flow_loss` 别名 |
| optimizer/scheduler | 相同 preset、LR、betas、eps、weight decay、clip 和 warmup/decay 逻辑；参数列表差异见下节 |
| 混合精度、分布式 | 同训练器、Accelerator 和 DDP 路径；精度和卡数必须另外匹配 |
| 推理 | 原 `sample_actions`、noise、Euler 步数、prefix KV cache、动作截维及后处理 |
| 动作执行 | 原 `predict_action_chunk`、`select_action`、队列、`n_action_steps` 和 episode reset |
| RTC / compile / PEFT | A 委托原实现；本次没有执行完整 RTC、编译器或 PEFT 集成测试；当前启动脚本关闭 PEFT |
| 模型与处理器保存加载 | 相同保存机制；测试验证 A 自身 roundtrip、旧 base 覆盖为 A、处理器状态及反归一化 |

## 仍保留的差异及其意义

1. `type/name` 是 `smolw`，config 保留 motion 和 TensorBoard 字段，日志多一个 action loss 别名。
   A 不使用 motion 字段。TensorBoard 默认 scalar 写入不参与计算图。
2. SmolW 的 `train_expert_only` **默认 False**，SmolVLA 配置类默认 True。
   你的旧 SmolVLA 脚本也显式设置 False，因此本次保留当前实验行为，并在 SmolW 启动脚本明确
   写出 False。现在 A 也接受显式 True；不能把“默认配置类”等同于“同配置实验”。
3. SmolW 的 `get_optim_params` 只返回可训练参数；SmolVLA 返回所有参数。冻结参数保持
   `grad=None` 时，AdamW 跳过它们，测试验证更新和每个有效参数的 optimizer state 一致。
   保留此布局以兼容已有 A checkpoint 的 optimizer resume。**原版与 A 的 optimizer 文件
   不应直接互换**；模型权重同键可比。中途手工解冻未进入优化器的参数需要重建优化器。
4. `forward` 保留 SmolW 的额外 `z_noise` 参数及参数顺序。训练器调用 `forward(batch)` 或
   使用 `reduction=`，没有差异；手写对照代码必须用 `noise=…`、`time=…`，不要将原版的
   `forward(batch, noise, time)` 位置参数照搬。A 另有 reduction 参数合法性校验。
5. `drop_n_last_frames` 是 SmolW 额外选项。当前脚本的 0 与原版相同；若改成正数，会改变
   训练起点集合，不能再称为相同数据条件的基线。
6. 当前 SmolW 的 RLDS 配置入口和异步 inference server 白名单不支持 `smolw`；原版 SmolVLA
   在这些入口有支持。当前 LeRobot 格式 LIBERO 训练及 `lerobot-eval` 不经过这些入口。

## 本次修正

- A 不再受旧 SmolW 的“必须联合训练 VLM”“必须满足 VidTwin 参数”“必须缓存”等配置限制；
  原 SmolVLA 的父类校验仍生效。A 不再创建无用历史队列。
- A 加载时允许忽略旧 base 额外的 motion 权重，但**缺少任意基础 SmolVLA 权重就报错**，
  不再只警告后使用随机初始化补齐。
- 转换器严格加载源权重；任意原始 tensor 无法复制、或基础 tensor 意外保留随机值时即报错。
- `test_smolw.sh` 原先默认指向旧联合目录 `smolw-union`，现在默认与训练脚本的 A/B/C/D
  输出目录对应。评估不覆盖 checkpoint 内保存的 `train_version`，仍可用 `POLICY_PATH`
  指定任意旧模型路径。episode 数等数值保持原样。
- 未修改 `z_loss_weight`、学习率、训练步数、batch size、chunk size、执行步数等实验数值；
  这些修正不要求已经正常训练的 A 仅因本次修改而重训。

## 与旧训练脚本的区别

| 设置 | 当前 `train_smolw_lr.sh` 的 A | `train_smolvla_lr.sh` |
| --- | --- | --- |
| 权重路径 | `models/smolw-base` | `models/smolvla` |
| chunk_size | 16 | 20 |
| n_action_steps | 10 | 20 |
| GPU / batch | 双卡，各 64，全局 128 | 单卡 128 |
| 精度 | 显式 Accelerate bf16 | 脚本未显式指定，需检查实际环境 |
| steps | 100000 | 60000 |
| VLM 训练 | expert_only=False，vision 冻结 | 相同 |
| LR / clip / warmup / decay | 3e-5 / 5 / 5000 / 100000 / 5e-6 | 相同 |
| dataset | 同一 root，repo_id=local | 同一 root，repo_id=libero_only |

双卡 64 与单卡 128 也不保证逐步等价。当前 loss 在每个 rank 内先按有效 action 元素归一化，
DDP 再平均梯度。当两个 rank 的有效元素数不同，`(S1/N1 + S2/N2)/2` 与单卡的
`(S1+S2)/(N1+N2)` 不同。这是历史实验配置差异，A 与原版在**相同双卡配置**下使用相同逻辑。

此外，转换器将 `load_vlm_weights=False`、`compile_model=False` 写入目标配置。
若源配置取值不同，对照原版时也需要显式匹配：即使最终权重相同，构造路径可能影响 dtype
或 RNG。冻结开关、features、tokenizer、normalization、use_amp、num_steps 等应比较
实际保存的配置及启动覆盖值，不能只比两份 shell 中出现的参数。

## 已执行的验证与边界

本次相关测试合计 **98 passed, 4 skipped**；Ruff、shell 语法及 `git diff --check` 通过。

新增数值测试使用小型真实 Llama 层，替换的是预训练模型构造、视觉编码和 tokenizer 下载。
执行的是原始 expert attention、RoPE、缓存、flow loss、动作采样及真实 optimizer。

- 同初值下参数、loss、所有梯度精确一致；包含 self/cross attention 和 mean/none。
- FP32 / CPU BF16，三组冻结设置，连续四步训练；覆盖 clipping、AdamW、warmup/scheduler、
  自身 optimizer resume。每步全部模型参数、有效 optimizer state、loss、梯度范数、LR
  以 `rtol=0, atol=0` 对比。
- 随机 noise/time 的 RNG 消耗一致；训练后含多次队列重填的随机动作推理逐元素一致。
- 多相机、缺失相机、部分相机无效、special image tokens、prefix padding，以及无/全/混合
  action padding；旧 processor 文件加载后的配置、归一化和反归一化一致。
- source → legacy base 全量权重复制、A 加载拒绝缺失权重、artifact 检查器识别权重/dtype/
  normalization/processor 配置损坏；ABCD 既有测试继续运行。

这不包含真实双卡训练、完整预训练视觉模型前向、视频解码、完整训练器端到端执行或 LIBERO
rollout，不能由这些测试推出某个成功率。当前本地缺少完整 dataset 依赖；训练器的数据及
update 路径是代码核对，数值更新测试独立调用真实 policy、Accelerator 和 optimizer。

## 服务器上验证已有 base

新增工具只读本地 artifact，每次比较一对 tensor，不加载 VLM 或 VidTwin：

```bash
python -m lerobot.policies.smolw.audit_smolvla_base \
  --source /data1/gaowenbing/WorkSpace/models/smolvla \
  --target /data1/gaowenbing/WorkSpace/models/smolw-base
```

应对比**训练前 base**，不要拿已经训练过的 A 与初始 SmolVLA 比权重。
`weights_and_processors_match=true` 表示原始 tensor 的 dtype/shape/值、处理器配置与统计一致
（仅忽略目标多出的 motion tensor 和会被 A 移除的 stationary 步骤）。非一致时退出码为 1。
同时列出共享配置差异；这个 true 只证明 artifact 转移，不能替代有效训练配置的匹配。

对应评估调用：

```bash
train_version=A CHECKPOINT_STEP=030000 bash test_smolw.sh
```

核对实际 `policy.path` 和 checkpoint `config.json` 内 `train_version=A`。比较原版成功率时，
还需相同任务集、相机映射、控制模式、初始状态/seed、episode 数和执行步数。
