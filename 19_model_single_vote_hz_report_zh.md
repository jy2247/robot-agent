# Robot Agent 模型配置、单投票机制与推理 Hz 报告

报告日期：2026-11-04（America/New_York）  
基准设计文档：[18_vlm_monitor_latency_engineering_zh.md](18_vlm_monitor_latency_engineering_zh.md)  
测试项目：`/data/jingyu/robot-agent`

## 1. 结论摘要

本项目当前实现的实时 monitor 是**真正的单投票**：一个新鲜时间窗口只发送一次模型请求，请求只允许返回一个 choice，不使用 `n=5`，也不在单次 assessment 内做多票并发。为降低单票误判风险，系统采用“确定性单票 + 两个连续新窗口给出相同高置信终态后接受”的时间确认机制。

本轮涉及四种模型/检查点：

| 模型 | 硬件/状态 | 任务协议 | 稳定态推理 Hz | 是否达到 1 Hz monitor | phase1 终态结果 |
|---|---|---:|---:|---:|---|
| Qwen3.8-27B-NVFP4 | A5000 SM86 不支持，未加载 | monitor 设计目标 | 无合法数据 | 否 | 未测试 |
| Qwen3.8-27B-AWQ | 2× RTX A5000，TP=2 | 单票 monitor | **0.485 Hz p50** | 否 | 9/9 有效，连续两票 success，9.75 s 接受 |
| Qwen2.5-VL-7B-Instruct-AWQ | 1× RTX A5000 | 单票 monitor | **1.221 Hz p50** | 基本达到；p95 略低于 1 Hz | 9/9 有效，仅最后一票 success，未满足两窗口确认 |
| Flora-TempSeg 9B | 1× RTX A5000 | 离线时间分段，不是 monitor verdict | **0.133 窗口/s 端到端**；纯生成 0.351 窗口/s | 不适用 | 输出整段 `[0, 10.433333]` |

因此：

1. **实时性首选 7B-AWQ**：单卡、p50 819.3 ms、1.221 Hz；但 p95 1011.0 ms，刚好超过 1 秒，而且当前 phase1 质量不足以触发两窗口终态接受。
2. **当前质量较好的是 27B-AWQ**：phase1 能正确形成两次连续 success 并接受，但单票约 2.064 秒，只能达到 0.485 Hz，且占用两张 A5000。
3. **Flora-TempSeg 应作为离线/低频阶段边界生成器**，不能直接替换实时 monitor。它不输出 `success/in_progress/fail + confidence`，也不判断 FC 指定的 `expected_phase`。
4. **NVFP4 不可在 A5000 上报告 Hz**。该检查点需要至少 SM89；当前 A5000 是 SM86。失败加载不能转换成 0 Hz，也不能用预估值代替实测。

## 2. 测试环境与口径

### 2.1 硬件和运行时

| 项目 | 配置 |
|---|---|
| GPU | 8× NVIDIA RTX A5000，单卡 23028 MiB，Compute Capability 8.6 |
| 实时 monitor 运行时 | Python 3.10.12、PyTorch 2.10.0+cu128、vLLM 0.19.1 |
| 27B-AWQ | GPU 0、1，tensor parallel 2 |
| 7B-AWQ | GPU 0，tensor parallel 1 |
| Flora-TempSeg | GPU 0，Transformers + PEFT，BF16 base、FP32 GDN/LoRA |

### 2.2 Hz 定义

报告区分以下指标，不能互相替代：

- `sequential_inference_hz = 1000 / p50_full_response_ms`：连续单票完整响应的 p50 频率，包含服务端视觉 prefill、decode、HTTP 返回和客户端 JSON 解析，不包含提前完成的视频解码/JPEG 编码。
- `offline_request_loop_hz`：整段离线 benchmark 中顺序请求数除以请求循环 wall time。
- `p95 保守 Hz = 1000 / p95_ms`：按 p95 延迟换算的保守频率。
- Flora `window_hz`：一个 16 秒/64 槽时间分段窗口从取帧、预处理、生成到 stitch 完成的频率。它不是 monitor 单票 Hz。
- Flora `generation_window_hz`：只计算模型 `generate()` 的窗口频率，不包含视频预处理和 stitch。

所有实时 monitor 数据来自真实 phase1 回放，不使用重复合成图；warmup 使用不同的 phase5 视频，避免把同一批视觉输入的 MM cache 命中当成真实性能。

## 3. 当前单投票机制

### 3.1 一次窗口只有一次模型请求

实现证据：

- `MonitorAgent.observe()` 每次只调用一次 `backend.generate(...)`。
- OpenAI backend 请求没有 `n` 字段，因此默认 `n=1`。
- backend 强制 `choices` 长度必须等于 1；返回多个 choice 会直接报错。
- benchmark 对每个 source timestamp 只调用一次 `agent.observe()`。

这与原设计文档 D7 的“5 个并发随机票”不同，是为 A5000 性能和低延迟做的明确架构变更。

### 3.2 单票稳定策略

当前参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `temperature` | 0.0 | 确定性生成，取消原文 `temperature=0.7` 的随机投票需求 |
| `top_p` | 1.0 | 不额外截断 nucleus |
| `terminal_confirmations` | 2 | 同一 phase、同一 terminal 状态必须在两个连续新窗口重复出现 |
| `confidence_threshold` | 0.70 | 低于阈值的 success/fail 不进入确认计数 |
| `assessment_hz` | 1.0 | 目标调度频率，不代表模型一定能够维持 1 Hz |
| request timeout | 8.0 s | 正式配置；benchmark 为避免误杀慢模型临时覆盖到 120 s |

确认状态在以下情况清空：

- 新窗口返回 `in_progress`；
- terminal 置信度低于 0.70；
- terminal 状态从 success 变为 fail 或相反；
- backend、传输或 JSON 解析失败。

backend 失败只回退为 `in_progress, confidence=0.0`，不会错误地当成物理失败。输入还强制 `frame_id` 唯一、timestamp 严格递增，从而保证“两票”来自两个不同的新窗口，而不是重复使用同一帧。

### 3.3 输出协议

实时 monitor 的 wire verdict 固定为：

```json
{"status":"in_progress|success|fail","confidence":0.0}
```

特点：

- 仅两个字段；
- 不输出 reason、evidence、Markdown 或 schema；
- 不使用 guided decoding / `response_format`；
- 客户端从回复中解析第一个合法的两字段 JSON，能容忍代码围栏或少量附加文字；
- status 和 confidence 都做严格值域校验；
- `max_tokens=32`，当前实测 completion median 为 13 tokens。

这对应设计文档 D1/D2 的核心要求。

## 4. Prompt 审核

当前 system prompt 的核心规则：

1. 只判断 controller 指定的 `ACTIVE_PHASE`，不自行推进 phase。
2. 按物体功能和初始工作区位置识别物体，不依赖固定颜色、标签、几何、视角或机器人手。
3. 只使用可见物理证据，不从意图或后续姿态反推未观察到的交互。
4. 可见成功条件一旦满足即可 success，不要求机器人静止。
5. 只有明确可见的 failure rule 才能 fail；遮挡、模糊、动作未完成或证据不足都返回 in_progress。
6. 多视角冲突时采用物理证据更清楚的视角，否则低置信 in_progress。
7. 严格输出两字段短 JSON。

每次请求只注入当前 phase 的四项 ACTIVE RULES：

```text
task
active_phase
success
failure
```

不会把五个 phase 的全部 rubric 都塞进热路径。静态 system + 当前 phase rubric 在前，变化的相机内容在后，符合 prefix cache 的前缀组织原则。

五个 phase 的 success/failure 规则已按工程约束收紧：

- step1：pipette 离开 holder 支撑且被 gripper 持有即可；不要求静止或额外位移。
- step2：目标 tube 离开 rack/初始支撑且被持有即可。
- step3：只要求 pipette tip 在同一被持 tube 的开口上方并对准；不要求接触、插入、按压或移液。
- step4：gripper 释放后，指定 rack pocket 稳定支撑同一 tube。
- step5：gripper 释放后，初始位置的 holder 稳定支撑同一 pipette。

### Prompt 与 18 号设计文档的对应关系

| 要求 | 当前状态 | 说明 |
|---|---|---|
| thinking 关闭 | 符合 | 每请求传 `chat_template_kwargs.enable_thinking=false` |
| 两字段短 JSON | 符合 | status + confidence，实测 median 13 tokens |
| 无 guided decoding | 符合 | 请求未传 `response_format`/JSON schema |
| 静态前缀在前 | 符合 | system + active rubric 在图像前 |
| 只判断当前 skill | 符合 | controller 提供 `expected_phase` |
| 不输出 reason | 符合 | prompt 明确禁止，wire schema 也不接受第三字段 |
| 多 system 合并 | 符合 | 当前只有一个 system message |
| 单票而非五票 | 有意变更 | `n=1`、temperature 0、两新窗口确认 |

## 5. 图像采样和请求参数审核

### 5.1 项目目标配置

| 参数 | 当前值 | 与设计文档关系 |
|---|---:|---|
| 相机数 | 配置目标 3；本次数据实测 2 | 部分符合；数据只有 head + left wrist |
| 每相机帧数 | 8 | 符合 |
| 帧间隔 | 0.25 s | 符合 |
| 窗口跨度 | 1.75 s（8 帧） | 与“最近约 2 秒”一致 |
| 分辨率 | 448×256 | 符合 |
| 图像编码 | JPEG quality 95，4:4:4 | 符合高质量、无色度子采样原则 |
| 顺序 | camera-major；每相机 oldest → newest | 符合 |
| thinking | false | 符合 |
| max tokens | 32 | 符合低延迟短输出目标 |
| temperature | 0.0 | 单票确定性配置；不同于原 5 票的 0.7 |
| top_p | 1.0 | 符合 |
| stream | false | 符合 |
| retries | 无自动 retry | 符合“一次窗口一次请求” |

### 5.2 当前与三相机生产目标的差距

本次 phase1 数据只有两个相机，因此每次请求为 16 张图，不是目标的 24 张图。两个 AWQ server 脚本当前也设置了：

```text
--limit-mm-per-prompt {"image":16}
```

而 `configs/monitor.toml` 的生产目标是 3×8=24 张。也就是说：

- 当前两相机 benchmark 是有效的两相机结果；
- 不能把当前 Hz 直接声明成三相机 24 图 Hz；
- 接三相机前必须把 server 的 image limit 调到 24，并重新测 p50/p95；
- 24 图视觉 token 增加后，Hz 会低于当前报告数字。

当前仍使用 16 个独立 `image_url`，尚未实现设计文档 D5 的“每路相机打包为一个 video 输入”。因此视频模式带来的 token 减半收益还没有进入本项目实测。

## 6. 各模型详细配置和实测

### 6.1 Qwen3.8-27B-NVFP4

| 项目 | 配置/结果 |
|---|---|
| 量化 | NVFP4 + FP8 compressed-tensors |
| 设计启动参数 | max model len 24576；max batched tokens 20480；max seqs 8；KV FP8；memory utilization 0.85；prefix cache；CUDA graph sizes `[1,2,4,8]` |
| 当前 GPU | RTX A5000，SM86 |
| 兼容性检查 | 失败：ModelOpt NVFP4 要求 Compute Capability ≥ 8.9 |
| 模型是否成功加载 | 否 |
| Hz | **不得报告** |

它最接近 18 号文档在 RTX 5090/SM120 上的原始方案，但不兼容当前 A5000。此结论是硬件 kernel 能力限制，不是 Prompt 或 agent 架构问题。

### 6.2 Qwen3.8-27B-AWQ

模型配置：

| 项目 | 值 |
|---|---|
| 架构 | `Qwen3_5ForConditionalGeneration` |
| 磁盘占用 | 18 GB |
| 量化 | AWQ 4-bit、group size 128、GEMM、zero point |
| vLLM kernel | AWQ Marlin |
| GPU | 2× RTX A5000 |
| tensor parallel | 2 |
| dtype | FP16 runtime |
| model load memory | 约 9.26 GiB/worker |
| max model len | 8192 |
| max batched tokens | 8192 |
| max seqs | 1 |
| GPU memory utilization | 0.88 |
| execution | enforce eager；custom all-reduce disabled |
| prefix cache | 开启 |
| generation config | `vllm`，避免检查点 generation config 覆盖请求 |
| MM encoder TP | data mode |

phase1 单票结果：

| 指标 | 实测 |
|---|---:|
| 请求数 / 有效数 / 错误数 | 9 / 9 / 0 |
| p50 单票完整响应 | 2063.9 ms |
| p95 单票完整响应 | 2156.6 ms |
| mean | 2065.4 ms |
| p50 Hz | **0.485 Hz** |
| p95 保守 Hz | **0.464 Hz** |
| 离线顺序循环 Hz | 0.484 Hz |
| prompt tokens median | 2229 |
| completion tokens median | 13 |
| 状态分布 | 7 in_progress、2 success |
| terminal 接受 | 1 次；第一个接受窗口 source time 9.75 s |

合规判断：

- 单投票、Prompt、thinking、JSON、采样参数：符合当前单票架构。
- 1 Hz 调度目标：不符合；一次响应约占两个 1 秒周期。
- 单卡目标：不符合；当前需要两张 A5000。
- 三相机目标：尚未验证；这里只测了两相机 16 图。
- phase1 时间确认：通过，能形成两个连续高置信 success。

### 6.3 Qwen2.5-VL-7B-Instruct-AWQ

模型配置：

| 项目 | 值 |
|---|---|
| 架构 | `Qwen2_5_VLForConditionalGeneration` |
| 磁盘占用 | 6.5 GB |
| 量化 | AWQ 4-bit、group size 128、GEMM、zero point；visual 不量化 |
| vLLM kernel | AWQ Marlin |
| GPU | 1× RTX A5000 |
| tensor parallel | 1 |
| dtype | FP16 runtime |
| 模型加载显存 | 6.57 GiB |
| max model len | 8192 |
| max batched tokens | 8192 |
| max seqs | 1 |
| GPU memory utilization | 0.90 |
| prefix cache | 开启 |
| generation config | `vllm` |
| image max pixels | 114688，正好覆盖 448×256 |

phase1 单票结果：

| 指标 | 实测 |
|---|---:|
| 请求数 / 有效数 / 错误数 | 9 / 9 / 0 |
| p50 单票完整响应 | 819.3 ms |
| p95 单票完整响应 | 1011.0 ms |
| mean | 844.0 ms |
| p50 Hz | **1.221 Hz** |
| p95 保守 Hz | **0.989 Hz** |
| 离线顺序循环 Hz | 1.184 Hz |
| prompt tokens median | 2710 |
| completion tokens median | 13 |
| prefix cache ratio | 0.12 |
| 状态分布 | 8 in_progress、1 success |
| terminal 接受 | 0；最后一帧才首次 success，缺少下一新窗口确认 |

合规判断：

- 单投票、Prompt、thinking、JSON、采样参数：符合当前单票架构。
- 1 Hz 目标：p50 达到；p95 轻微未达到，差约 11 ms。
- 单卡目标：符合。
- 三相机目标：尚未验证；这里只测了两相机 16 图。
- phase1 时间确认：未通过。问题不是请求错误，而是视频结束前只产生一票 success。

### 6.4 Flora-TempSeg 9B

模型配置：

| 项目 | 值 |
|---|---|
| 模型形式 | Qwen3.5-9B 精确 revision + Flora LoRA adapter |
| adapter | rank 32、alpha 64、约 351 MB 权重 |
| 精度 | BF16 base、FP32 Gated DeltaNet、FP32 LoRA |
| GPU | 1× RTX A5000 |
| 输入 | 单相机；16 s envelope；4 Hz；64 temporal slots |
| Prompt | Flora 官方 temporal segmentation prompt |
| thinking | false |
| 解码 | greedy，`do_sample=false`、`num_beams=1` |
| max new tokens | 1024 |
| 输出 | `{"segments":[{"s","e","lo","ro"}]}`，20 Hz local ticks |
| 常驻显存 | 17.865 GiB |
| 峰值显存 | 18.353 GiB |

phase1 结果：

| 指标 | 实测 |
|---|---:|
| 冷加载 | 62.399 s |
| warmup 生成 | 3.642 s |
| measured 纯生成 | 2.852 s，**0.351 窗口/s** |
| measured 窗口端到端 | 7.500 s，**0.133 窗口/s** |
| 视频处理倍率 | 1.391× realtime |
| 输出 tokens | 24 |
| 输入 tokens | 1811 |
| 原始输出 | `{"segments":[{"s":40,"e":249,"lo":false,"ro":false}]}` |
| stitch 后区间 | `[0.0, 10.433333]` s |

它不对应实时 monitor 的单票协议：

- 不接收 FC 的 `expected_phase` success/failure rubric；
- 不输出 status/confidence；
- 一个请求处理固定 16 秒 envelope，而不是最近 1.75 秒监控窗口；
- 输出是无动作标签的时间区间，适合视频预分段或离线标注；
- 当前数据只有主相机和左腕两路，而 Flora 原生只支持 1 路或固定 3 路，因此本次只用了主相机。

因此 Flora 的 0.133/0.351 是“分段窗口 Hz”，不能和 27B/7B 的 monitor 单票 Hz 直接排名。

## 7. 与 18 号工程文档逐项对照

| 设计项 | 当前实现 | 判定 |
|---|---|---|
| D1 thinking off | 每请求 `enable_thinking=false` | 符合 |
| D2 两字段短 JSON | 严格 status/confidence，max 32 tokens | 符合 |
| D2 不用 guided decoding | 未使用 response_format/schema | 符合 |
| D3 本地量化部署 | AWQ Marlin 本地部署成功 | 符合；量化类型由 NVFP4 改为 AWQ |
| D3 NVFP4 | A5000 SM86 不支持 | 不符合当前硬件 |
| D4 448×256、3×8、0.25 s | 尺寸/帧数/间隔符合；本次只有 2×8 | 部分符合 |
| D5 video 打包 | 当前仍是独立 image_url | 未实现 |
| D6 prefix cache | server 开启，Prompt 静态前缀在前 | 符合；当前实际收益有限 |
| D7 五票并发 | 改为一次请求单票 | 有意变更 |
| 单票稳定性 | temperature 0 + 两新窗口确认 | 已实现 |
| D8 1 s 节拍 | 配置为 1 Hz | 配置符合；27B 算力不满足，7B 基本满足 |
| D8 backend 错误处理 | 回退 in_progress，不当 fail | 符合 |
| D9 server 5090 参数 | A5000/AWQ 使用 8192、seqs 1、不同显存参数 | 按硬件重配，不是原参数复现 |
| D10 temperature/top_p | 0.0/1.0 | 与单票架构一致，不同于五票 0.7 |
| D10 stream/logprobs | stream false，无 logprobs | 符合 |
| 真实图片 benchmark | phase1 真实视频，warmup 使用不同视频 | 符合 |
| TTFT 独立测量 | 当前 summary 只有完整响应延迟 | 尚缺 |
| snapshot→接受端到端 | 离线回放有 source time 和接受点，但没有真实相机 wall-clock | 尚缺生产实测 |

## 8. 推理 Hz 总表

| 模型 | 相机×帧 | p50 latency | p95 latency | p50 Hz | p95 保守 Hz | loop/window Hz | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.8-27B-NVFP4 | 未运行 | — | — | — | — | — | SM86 不兼容，禁止填 0 Hz |
| Qwen3.8-27B-AWQ | 2×8 | 2063.9 ms | 2156.6 ms | **0.485** | **0.464** | 0.484 | 2×A5000，终态接受成功 |
| Qwen2.5-VL-7B-AWQ | 2×8 | 819.3 ms | 1011.0 ms | **1.221** | **0.989** | 1.184 | 1×A5000，未形成两窗口终态确认 |
| Flora-TempSeg 9B | 1×64 | 7500 ms measured E2E | 单次样本，无 p95 | 不可比 | 不可比 | **0.133 window Hz** | 纯生成 2.852 s / 0.351 window Hz |

相对速度：7B-AWQ 的 p50 单票频率约为 27B-AWQ 的 **2.52 倍**。

## 9. 当前推荐配置

### 9.1 在线实时 monitor

优先采用 Qwen2.5-VL-7B-Instruct-AWQ 单卡配置，保留：

```text
n=1
temperature=0.0
top_p=1.0
max_tokens=32
enable_thinking=false
timeout_s=8
terminal_confirmations=2
confidence_threshold=0.70
8 frames/camera
0.25 s spacing
448×256
```

但上线前必须补三项验证：

1. 三相机 24 图实测，并把 server `limit-mm-per-prompt` 从 16 调到 24。
2. 在包含 phase 完成后额外稳定画面的录像上验证两窗口确认，避免视频结束时只有一票 success。
3. 测 TTFT、完整单票和真实 snapshot→accepted 三种延迟，不再只看完整响应 p50。

如果质量优先且允许低频，可使用 27B-AWQ，把 scheduler 频率调整到约 0.45 Hz，或接受一次请求跨越两个 1 秒周期；不能继续宣称它满足 1 Hz。

### 9.2 离线阶段切分

Flora-TempSeg 保留为独立的离线预处理器：先对完整五阶段视频产生候选边界，再由实时 monitor 或人工复核赋予 phase 标签。不要把 Flora 输出直接接到 `MonitorAgent.monitor_verdict()`。

## 10. 可复查证据

- 实时 monitor 配置：`configs/monitor.toml`
- monitor Prompt：`src/robot_agent/prompts/monitor.txt`
- Prompt 构造：`src/robot_agent/context.py`
- 单票 backend：`src/robot_agent/backends/openai.py`
- 两窗口确认逻辑：`src/robot_agent/agent.py`
- 27B-AWQ summary：`outputs/benchmark/phase1-awq-a5000-measured/phase1/summary.json`
- 27B-AWQ vLLM log：`outputs/benchmark/phase1-awq-a5000-measured/vllm.log`
- 7B-AWQ summary：`outputs/benchmark/phase1-qwen25-vl-7b-awq-single-gpu-measured/phase1/summary.json`
- 7B-AWQ vLLM log：`outputs/benchmark/phase1-qwen25-vl-7b-awq-single-gpu-measured/vllm.log`
- NVFP4 兼容性诊断：`outputs/diagnostics/nvfp4_runtime.json`
- AWQ 兼容性诊断：`outputs/diagnostics/awq_runtime.json`
- Flora 结果：`outputs/benchmark/phase1-flora-tempseg/result.json`
- Flora log：`outputs/benchmark/phase1-flora-tempseg/run.log`

## 11. 报告限制

- 两个实时 monitor benchmark 都只使用 2 路相机、16 张图，尚不是三相机生产配置。
- 每个实时模型只有一个 phase、9 个 measured windows；p50/p95 可用于当前工程比较，但样本量不足以给出稳定的生产 p99。
- 当前没有标注级 ground truth 边界和逐窗口准确率，因此本报告只说明输出行为和 terminal acceptance，不宣称总体准确率。
- Flora 只有一个 measured window，没有 p50/p95 分布；报告仅给出该次稳定态 measured run。
- 输入预处理在实时 monitor benchmark 中提前完成，没有计入 `inference_and_parse_ms`；真实在线 snapshot 端到端延迟会略高。
