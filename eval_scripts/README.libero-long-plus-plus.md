# 用 LeRobot checkpoint 评估 LIBERO-Long++

在 LeRobot 仓库根目录运行以下命令。已有 LiLo-VLA 仿真环境时，可通过 `--benchmark-python /path/to/bin/python` 指定解释器，通过 `--benchmark-root /path/to/LiLo-VLA` 指定 checkout。

```bash
# 列出默认套件的任务编号
uv run --no-sync python eval_scripts/run_libero_long_plus_plus.py --list-tasks

# 跑一个短 episode，验证模型、仿真和动作接口
uv run --no-sync python eval_scripts/run_libero_long_plus_plus.py \
  --checkpoint /path/to/pretrained_model --device cuda \
  --task-id 0 --episodes 1 --max-steps 20

# 完整评估，默认 12 个任务配置 × 10 个 episode
uv run --no-sync python eval_scripts/run_libero_long_plus_plus.py \
  --checkpoint /path/to/pretrained_model --device cuda
```

`--checkpoint` 也接受包含 `pretrained_model` 的训练 checkpoint 目录或 Hugging Face repo ID。使用训练该模型时的 LeRobot 环境和模型依赖。脚本读取 checkpoint 的预处理、后处理和动作块配置，自动识别 `image` / `image2` 和 `base_0_rgb` / `left_wrist_0_rgb` 两套相机命名。其他命名可用 `--agent-camera` 和 `--wrist-camera` 指定。动作控制默认 `relative`，也支持 `--control-mode absolute`。PI05-LAP 需要能预测机器人动作的 post-training checkpoint。

默认每个 episode 最多执行 520 步模型动作，控制频率为 20 Hz。策略在整个 episode 接收同一条完整任务指令。每局通过 LiLo-VLA 随机 reset，沿用其 20 步仿真稳定阶段，之后逐步检查技能谓词。首次乱序时结束该局。同一步触发的连续技能按目标顺序计入。`eval_info.json` 每局更新，包含逐局结果、逐任务和总体的成功率（SR）与平均进度（AP）。SR 同时要求按序完成技能且最终 BDDL 目标成立，AP 是乱序或超时前已按序完成的技能比例。默认输出到 `outputs/eval/libero_long_plus_plus/` 下的时间戳目录。

## 首次安装仿真环境

LiLo-VLA 的仿真依赖使用自己的 Python 环境。以下命令默认从当前仓库旁边的 `benchmarks/LiLo-VLA` 查找 checkout。安装 `egl-probe` 需要 PATH 上有 CMake 和 C++ 编译器，CMake 4 可用 `CMAKE_POLICY_VERSION_MINIMUM=3.5` 构建此依赖。

```bash
LILO="$(cd ../benchmarks/LiLo-VLA && pwd)"
LIBERO_REV=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
LIBERO_SRC="$LILO/.venv/src/LIBERO-$LIBERO_REV"
uv venv --python 3.11 "$LILO/.venv"
mkdir -p "$LILO/.venv/src"
curl -fL "https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/$LIBERO_REV" \
  | tar -xz -C "$LILO/.venv/src"
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install --python "$LILO/.venv/bin/python" \
  -e "$LILO" 'future==0.18.2' 'gym==0.25.2' matplotlib 'opencv-python==4.11.0.86'
printf '%s\n' "$LIBERO_SRC" > "$LILO/.venv/lib/python3.11/site-packages/lilo_libero.pth"
```

`.pth` 让仿真环境从固定版本的 LIBERO 源码加载 Python 模块和仿真资产。Linux 默认用 `MUJOCO_GL=egl`，可用 `MUJOCO_GL=osmesa` 切换到已安装的 Mesa 软件渲染器。`--suite libero_long_plus_plus` 选择原顺序，`--suite libero_long_plus_plus_variant` 选择重排版，两者均使用各自从 0 开始的任务编号。
