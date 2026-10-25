# Robot Observer Agent

面向 **Flow Conductor（FC）+ VLA** 的时序视觉观察/critic 基础项目。
输入当前和历史观测，输出阶段、进展、错误和证据，由 FC 决定后续操作。
当前实现包含本地 Qwen3.8-27B/vLLM 部署、离线双视角回放和 FC 开发接口；真实相机与机器人控制仍需接入。

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

## 一键视频推理

本地模型权重和 `.venv-model` 准备好后，只需给出一个 MP4 或 episode 目录：

```bash
./run_agent.sh /path/to/video.mp4
./run_agent.sh /path/to/episode_directory
```

启动器会检测本地 `127.0.0.1:8000` 的 vLLM；没有服务时自动启动
Qwen3.8-27B，处理完后只关闭本次自己启动的服务。本地服务不需要 API key。
默认产出带 phase/status/decision/error/evidence 面板的 MP4，以及逐帧结果和汇总 JSON：

```text
video_agent.mp4
video_agent.predictions.json
video_agent.summary.json
```

对 Vega 失败数据集原始 episode 目录，脚本可以直接读取 `annotation.json` 和
`cams/*_view.mp4`，自动选择视角、任务阶段和失败时间窗。对单个视频，可以显式
给出阶段和额外同步视角：

```bash
./run_agent.sh ego.mp4 --view left_wrist.mp4 --view side.mp4 \
  --phase step4_tube_return --output outputs/my_result.mp4
```

可用阶段为 `step1_pipette_pick`、`step2_tube_pick`、
`step3_pipette_aim_at_tube`、`step4_tube_return` 和 `step5_pipette_return`。
`--sample-hz` 控制模型抽帧频率，输出视频仍保留原视频帧率。

可复现的十条随机 phase 拼接批处理：

```bash
./run_random_phase_batch.sh
```

默认随机种子为 `20261020`，产出 5 条成功数据集的完整五阶段链，以及 5 条
“成功阶段前缀 + 失败数据集的失败阶段”链。每条链有独立文件夹，包含
`result.mp4`、`predictions.json`、`inference_times.jsonl`、`source_manifest.json` 和
`summary.json`。数据集标签不进入模型请求；视频面板按时间显示最近一次原始模型输出。
可通过 `SEED=123 ./run_random_phase_batch.sh outputs/my-random-batch` 使用新的随机抽样。

## 项目结构

| 路径 | 用途 |
|---|---|
| `src/robot_agent/agent.py` | 观察循环、逐阶段累计确认、任务完成与错误处理 |
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
输入包括 `episode_id`、`frame_id`、`timestamp_s`，可选 `expected_phase`、`end_of_episode`、
`images`（PNG/JPEG base64 data URL 列表）和 `note`。
`expected_phase` 是 FC 当前要求执行的阶段，critic 仍须用图像验证，不能把它当作成功证据。
这里的 timestamp 是观测源时间；必须保持同一时间基准。
输出包含 `phase/status/confidence/errors/evidence`、阶段确认、任务完成、`decision`、recommendation 和延迟。
`confidence` 是当前 `phase+status` 判断的模型自评，不是整条任务的成功概率，尚未校准。
新 phase 开始时允许 `uncertainty_patience` 个低置信度观测继续收集证据；持续低置信度才转为
`review`。低置信度不能确认完成，任何失败判断都不使用该宽限。推荐结果是建议，FC 才负责实际控制。

移液任务的三份配置现在共用功能识别判据：第三步只需移液器尖端对准所持试管口上方，
不要求接触、插入或按压；此阶段表示对准完成，不证明实际移液完成。
第四、五步必须验证松手后的稳定支撑。阶段完成独立累计；看不到之前的动作不会清空记录，
进入下一阶段也不能补证前一步成功。明确的高置信度失败会撤销相应阶段的完成状态。
第一步是移液器已被夹爪拿住、脱离支架支撑；第二步是试管已被拿住、脱离架子支撑。
前三步条件满足即输出完成，继续移动不会使已满足的条件变成“进行中”，无需等待静止。
程序按阶段累计高置信度完成判断（普通配置仍需两次），不再要求连续。阶段切换、遮挡和
后端错误不清零；明确失败清除对应阶段的待确认计数。`phase_confirmation_counts` 可用于诊断。

episode 结束时发送 `end_of_episode=true`（也接受 `is_terminal=true`，同时提供时必须一致）。
程序输出 `overall_status`、`overall_reason`、`completed_phases` 和 `failed_phases`：
全部完成且无未解决失败才返回整体成功；否则 `decision=fail`，原因是 `PHASE_FAILED`
或 `INCOMPLETE_AT_END`。视觉 `status` 可以仍是 `uncertain`，这不等于已经观察到物理失败。
终止后必须 reset 才能继续输入。普通视频入口默认按 2 Hz 抽帧，只有读到源视频末尾才标记终止，
截取中间片段的结束不等于整个 episode 结束。抽帧频率不代表模型实际推理吞吐。

失败错误码按恢复成本映射：`missed_grasp/misaligned/short_of_target/off_target_press/stalled`
建议 `retry`；物体掉落、倾倒、位置错误或夹具位移建议 `replan`；危险运动、物体丢失、硬件故障或
不可恢复状态建议 `fail`。`WorkflowTracker` 只跨 episode 保存已接受阶段和重试次数，不保存图像；
乱序阶段会被拒绝，重试超过 `max_retries` 后升级为 `fail`。

## 连接服务器上的 VLM

先在独立环境启动支持图像输入的模型服务，再配置：

```bash
export ROBOT_VLM_BASE_URL=http://127.0.0.1:8000/v1
export ROBOT_VLM_MODEL=your-actual-served-model-name
# 如服务需要密钥：export ROBOT_VLM_API_KEY=...
robot-agent --backend openai serve
```

模型名称没有写死；请使用实际部署成功的视觉模型 ID/服务别名。本地 vLLM 默认不需要 API key；
`ROBOT_VLM_API_KEY` 只用于启用了鉴权的远程或自定义服务。
客户端使用 `/v1/chat/completions` 和 `image_url` 多模态格式；接口依据
[vLLM 官方文档](https://github.com/vllm-project/vllm/blob/main/docs/serving/openai_compatible_server.md)。
真实服务的模型支持、模板、多图上限和 JSON 输出能力必须实际验证。模型环境与轻量 agent
环境分开安装；本仓库已在 4×A5000 上验证 Qwen3.8-27B BF16 + vLLM 0.31。

## 已实现与后续边界

- 已实现：mock 回放、真实服务客户端、有限历史缓存、JSON 校验、逐阶段累计完成确认、开发 HTTP 接口、逐帧延迟日志、vLLM 启动脚本和视频 demo。
- 尚未实现：相机/ROS2 适配、FC 真实协议适配、自动重规划和模型微调。
- 3–5 Hz 是研究目标。默认预算 3 Hz；当前串行处理超时可能远高于预算。
- KV cache 由推理服务管理；这里仅保持稳定提示前缀。滑动历史和新图像可能降低缓存命中率。
- HTTPServer 是本地开发适配器，无认证、并发会话或生产部署能力；默认仅监听 loopback。

详见 [服务器部署](docs/server_setup.md)、[架构](docs/architecture.md)、[后续路线](docs/roadmap.md)。

## Qwen baseline 与离线评测

具体下载和服务命令见 [Qwen 部署](docs/server_setup.md#qwen38-27b-baseline)。
`configs/qwen38.toml` 单独连接 Qwen3.8-27B，`configs/g1_pipette.toml` 用于 G1 吸液管任务。

```bash
PYTHONPATH=src python -m robot_agent evaluate --predictions outputs/demo.jsonl --labels data/labels.jsonl
```

标签为 JSONL，每行包含 `episode_id/frame_id/phase/status`；与回放输出按 episode + frame
严格一一匹配，缺失、重复、多余帧均报错。输出阶段 accuracy、macro-F1（配置内全部阶段，
无样本阶段计 0）、混淆矩阵、逐帧完成/失败 precision 与 recall、agent 延迟最近秩 P50/P95、
预算超时率及 backend 错误率。分母为零的 precision/recall 输出 null。
完成命中必须阶段也正确。评测保留 backend 名称，mock 指标仅说明管道可用。
这里尚未计算事件级检测延迟或机器人闭环收益。

指定数据集的准备、逐 episode reset、双视角 640px 抽帧和评测命令见
[G1 数据集与 Qwen3.8 部署](docs/server_setup.md#g1-pipette-data-and-evaluation)。

## 成功/失败视频 demo

本地 Qwen 服务 ready 后执行：

```bash
PYTHONPATH=src python scripts/demo_pipette.py
```

脚本按需下载成功集第 0 个 episode，以及失败集 `ws_a2mem_chain_b0065` 的 ego/side 视频，
保留五个时刻的上下文并输出到 `outputs/pipette-demo/summary.json`。成功片段应接受
`step1_pipette_pick` 并推进到第 2 阶段；失败片段的标注是 S6 tube drop，映射到
`step4_tube_return`，应保持该阶段并建议 `replan`。Vega 失败集是仿真 TEST 集，不能与真实 G1
成功集混合训练后再声称同布局 held-out 表现。

## 五阶段可视化 demo

`scripts/demo_workflow_video.py` 使用成功数据集中同一 session 的 episode 237–241，按 step1–step5
拼成一个 51 秒视频。FC 为每段提供目标阶段，VLM 用时间上下文和可用双视角验证；
`WorkflowTracker` 只接受有序 transition，右下角绿灯来自模型实际的 `success` 输出。

```bash
bash scripts/serve_qwen.sh
PYTHONPATH=src .venv-model/bin/python scripts/demo_workflow_video.py \
  --sample-hz 0.5 --image-side 448
```

输出位于 `outputs/workflow-demo/five_phase_critic_demo.mp4`、`predictions.json`、`summary.json`
和 `vllm-benchmark.json`。本机实测接受 5/5 阶段；双视角时序链平均 3889 ms/次（0.257 Hz）。
这仍未达到 3–5 Hz，下一轮应比较 AWQ/FP8 量化、小型 VLM或快检测器配合慢 critic。

## 多类型失败 demo

下面的脚本从 Vega 失败集中下载五个短片，按 FC 当前阶段选择 ego、side、left-wrist 或
right-thumb 视角，输出每例独立视频和串联总览。数据集标注不会进入模型提示词。

```bash
PYTHONPATH=src .venv-model/bin/python scripts/demo_failure_videos.py --image-side 448
```

输出位于 `outputs/failure-demos/`。Vega 使用独立配置，明确追踪黄色试管、绿色吸头和白蓝支架；
当前严格命中 2/5：放回时试管掉落正确输出 `object_dropped/replan`，按压不足输出
`stalled/retry`。另外三例分别被判为进行中、错误类型不符和假成功。Vega 的 failure kind 和
可见起点来自状态/评测器推导，`human_verified=false`，这组结果用于集成诊断，不能作为正式
失败检测准确率。
