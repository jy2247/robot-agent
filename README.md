# Robot Observer Agent

面向 **Flow Conductor（FC）+ VLA** 的时序视觉观察/critic 基础项目。
输入当前和历史观测，输出阶段、进展、错误和证据，由 FC 决定后续操作。
目前是可运行的开发骨架，真实相机、模型部署、机器人控制及评测数据需要后续接入。

## 快速开始

Python 3.11+；mock 和 HTTP 客户端只用 Python 标准库，无需 GPU。
在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
robot-agent replay
python -m unittest discover -s tests -v
robot-agent serve
```

不安装包也能运行：`PYTHONPATH=src python -m robot_agent replay`。
回放结果写入 `outputs/demo.jsonl`。示例的 note 标签仅用于 mock，不代表视觉推理效果。

## 项目结构

| 路径 | 用途 |
|---|---|
| `src/robot_agent/agent.py` | 观察循环、连续确认、任务完成与错误处理 |
| `src/robot_agent/context.py` | 稳定前缀、历史图像和当前观测组装 |
| `src/robot_agent/backends/` | Backend 协议、mock、OpenAI-compatible 客户端 |
| `src/robot_agent/schemas.py` | 输入和输出校验 |
| `src/robot_agent/prompts/` | 阶段识别、成功/失败判断提示词 |
| `src/robot_agent/server.py` | FC 开发用 HTTP 接口 |
| `configs/default.toml` | 五阶段示例、缓存长度、模型与延迟预算 |
| `examples/` | mock 回放和请求示例 |
| `tests/` | 状态、接口、模型传输和异常回退测试 |
| `docs/` | 架构、部署、评测及下一步任务 |
| `AGENTS.md` | 服务器上 Codex 的开发上下文 |

## FC 接口

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/observe \
  -H 'Content-Type: application/json' --data-binary @examples/observation.json
curl -X POST http://127.0.0.1:8080/reset \
  -H 'Content-Type: application/json' -d '{}'
```

每个服务实例只处理一个 episode；新 episode 前显式 reset。同一 episode 的时间戳严格递增。
输入包括 `episode_id`、`frame_id`、`timestamp_s`，可选 `images`（PNG/JPEG base64 data URL 列表）和 `note`。
这里的 timestamp 是观测源时间；必须保持同一时间基准。
输出包含 `phase/status/confidence/errors/evidence`、阶段确认、任务完成、recommendation 和延迟。
`confidence` 是模型自评，尚未校准。推荐结果是建议，FC 才负责实际控制。

## 连接服务器上的 VLM

先在独立环境启动支持图像输入的模型服务，再配置：

```bash
export ROBOT_VLM_BASE_URL=http://127.0.0.1:8000/v1
export ROBOT_VLM_MODEL=your-actual-served-model-name
# 如服务需要密钥：export ROBOT_VLM_API_KEY=...
robot-agent --backend openai serve
```

模型名称没有写死；请使用实际部署成功的视觉模型 ID/服务别名。
客户端使用 `/v1/chat/completions` 和 `image_url` 多模态格式；接口依据
[vLLM 官方文档](https://github.com/vllm-project/vllm/blob/main/docs/serving/openai_compatible_server.md)。
真实服务的模型支持、模板、多图上限和 JSON 输出能力必须实际验证。
本包不安装 torch/vLLM，也不宣称已经验证任何具体模型。

## 已实现与后续边界

- 已实现：mock 回放、真实服务客户端、有限历史缓存、JSON 校验、连续完成确认、开发 HTTP 接口、逐帧延迟日志。
- 尚未实现：相机/ROS2 适配、FC 真实协议适配、自动重规划、模型微调、GPU 部署脚本、标注评测集。
- 3–5 Hz 是研究目标。默认预算 3 Hz；当前串行处理超时可能远高于预算。
- KV cache 由推理服务管理；这里仅保持稳定提示前缀。滑动历史和新图像可能降低缓存命中率。
- HTTPServer 是本地开发适配器，无认证、并发会话或生产部署能力；默认仅监听 loopback。

详见 [服务器部署](docs/server_setup.md)、[架构](docs/architecture.md)、[后续路线](docs/roadmap.md)。
