# 在实验室服务器继续开发

## 从 GitHub clone

私有仓库：`jy2247/robot-agent`。服务器需先具备该账号的 GitHub 访问权限：

```bash
git clone https://github.com/jy2247/robot-agent.git
cd robot-agent
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
robot-agent replay
python -m unittest discover -s tests -v
```

需要 Python 3.11+。若系统 Python 版本较旧，使用服务器已有的 Python 模块或独立环境。
包本身不要求 CUDA；先跑通 mock，再接模型服务。

## 尚未发布时使用源码包

解压后在 `robot-agent` 根目录执行同样的安装和运行命令即可。
如自行发布，先在自己的 GitHub 创建空仓库，再执行：

```bash
git init -b main
git add .
git commit -m "Initialize temporal VLM observer scaffold"
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## 模型环境

1. 检查 `nvidia-smi`、GPU 空闲显存、驱动和当前软件环境。
2. 按最终选定模型的官方说明，在独立环境部署支持多图输入的服务。
3. 确认真实模型服务名称、端口、chat template、多图数限制和显存消耗。
4. 导出 README 中的 ROBOT_VLM 环境变量，运行 `robot-agent --backend openai serve`。
5. 先用一帧真实图像测试，再增加历史长度，测量 P50/P95 延迟和判断质量。

没有预设特定 Qwen 型号、量化格式或 GPU 数；这些需在服务器根据真实模型确认。
模型服务与 agent 可在同一服务器不同环境运行。

## 让 Codex 接续

在服务器上打开项目根目录，让 Codex 阅读 `AGENTS.md`、README 和 `docs/roadmap.md`。
可以直接给它以下任务：

> 阅读 AGENTS.md，先运行 mock 回放和测试。检查本机 GPU 与现有模型服务，
> 根据我提供的真实 endpoint/model name 接通 OpenAIBackend。
> 使用真实相机帧验证阶段识别和错误回退，记录延迟。
> 保留 CPU mock 路径，不把 3–5 Hz 写成已达到的结果。

该仓库已提供供 Codex 接续的文件和任务上下文；Codex 本身的远端安装/登录需在服务器完成。

## Qwen3.8-27B baseline

本仓库使用官方视觉模型 `Qwen/Qwen3.8-27B`。模型卡确认其支持图像和视频输入，BF16
权重约 55.6 GB；A5000 需要多卡张量并行。
[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)。下载目录被 git 忽略；下载可续传：

```bash
mkdir -p .cache/huggingface .tmp
HF_HOME="$PWD/.cache/huggingface" HF_XET_CACHE="$PWD/.cache/huggingface/xet" \
TMPDIR="$PWD/.tmp" python scripts/download_model.py
```

把 Hugging Face 缓存和临时目录放在有空间的数据盘；系统 `/tmp` 所在分区满时，默认缓存会导致下载失败。
`huggingface_hub` 仅下载环境需要，不属于 agent 的运行依赖。
模型服务使用独立环境，避免修改现有训练环境：

```bash
python -m venv .venv-model
TMPDIR="$PWD/.tmp" PIP_CACHE_DIR="$PWD/.cache/pip" .venv-model/bin/pip install -U vllm
bash scripts/serve_qwen.sh
```

模型卡提供 vLLM OpenAI-compatible serving 示例；实际启动仍须检查本机 vLLM 对当前模型架构的支持。
安装后记录 `pip freeze` 和实际启动日志，不能仅凭安装成功认定模型可运行。
脚本默认 GPU 0–3、TP=4、BF16、8K 上下文、每请求最多 8 张图、最多 4 个序列，
并启用 prefix cache、chunked prefill 和 async scheduling。可通过 `CUDA_VISIBLE_DEVICES`、
`TP_SIZE`、`MAX_MODEL_LEN`、`MAX_NUM_SEQS` 和 `MM_LIMIT` 调整。
先检查 GPU 空闲状态。下载完全结束后才启动服务。

```bash
PYTHONPATH=src python -m robot_agent --config configs/qwen38.toml observe-image /path/to/frame.png
PYTHONPATH=src python -m robot_agent --config configs/qwen38.toml serve
```

专用配置通过 `chat_template_kwargs.enable_thinking=false` 和 `top_k=20` 关闭冗长思考，
为短 JSON 响应留足 token 预算；
其他模型配置默认不发送这一参数。空响应、截断、无效 JSON 仍按 uncertain 处理。
`observe-image` 的模型错误返回非零退出码；单帧 smoke 只验证集成，不能验证任务完成率。

## G1 pipette data and evaluation

数据源为 [jren313/g1-pipette-2view-teleop0925-5task-eerel](https://huggingface.co/datasets/jren313/g1-pipette-2view-teleop0925-5task-eerel)。
数据集卡列出 447 个 episode、272,236 帧、60 fps、两个 1280×720/1080×1080 视觉流，覆盖五个有序工作步骤。
LeRobot 会按需下载数据到 `$HF_LEROBOT_HOME`；将该目录指向空间充足的数据盘，然后创建独立数据工具环境并安装 `lerobot`：

```bash
python -m venv .venv-data
TMPDIR="$PWD/.tmp" PIP_CACHE_DIR="$PWD/.cache/pip" .venv-data/bin/pip install lerobot
HF_HOME="$PWD/.cache/huggingface" HF_LEROBOT_HOME="$PWD/data/lerobot" \
TMPDIR="$PWD/.tmp" PYTHONPATH=src .venv-data/bin/python scripts/evaluate_g1.py --max-episodes 2
```

小样本通过后删除 `--max-episodes 2` 可评测全部 episode。脚本每个 episode 前 reset agent，
按 `--sample-hz` 抽帧（默认 1 Hz），只发 head 与 left-wrist 两个视角，并缩到最长边 640px、JPEG 82 quality，
再调用本地 Qwen3.8 服务。输出帧级 predictions、labels 和 report 到 `outputs/g1/`；
报告用于阶段识别、完成检测、延迟和 deadline miss 分析。

该数据集主要包含成功演示，被标记失败的原始 episode 已从发布集剔除；所以它能提供阶段与阶段结束标签，
不能用于估算失败检测 recall 或 precision。终端阶段标签来自数据集人工检查过的切分边界，并把最后两个采样点作为完成确认窗口；
这是离线代理标签，仍需在真实相机流上校准。任务需要按源 session 分组留出，不能随机拆相邻 episode 后声称泛化。
完整数据集基准会串行调用模型数千次，先用两集验证环境和模板，再启动完整评测。

五阶段视频 demo 使用 episode 237–241 的连续 step1–step5 切片：

```bash
PYTHONPATH=src .venv-model/bin/python scripts/demo_workflow_video.py \
  --sample-hz 0.5 --image-side 448
```

脚本将腕部相机旋转为正向，并在 tube pick、aim 和 tube return 阶段与头部视角联合判断；
每个 episode 使用独立的单阶段 critic，跨 episode 只由 `WorkflowTracker` 保存阶段状态。
2026-10-20 在 4×A5000、Qwen3.8-27B BF16、vLLM 0.31 上的演示结果为 5/5 阶段按序接受，
完整双视角链平均 3889 ms/次（0.257 Hz）。单视角端点基准约 2.7–3.7 秒/次；由于样本少且
服务 warm-up/生成长度有波动，分辨率间差异不能当成稳定吞吐结论。
