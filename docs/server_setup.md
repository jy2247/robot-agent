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
