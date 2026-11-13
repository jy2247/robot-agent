# Robot VLM Monitor

## B200 NVFP4：单票与五票 benchmark

测试环境：单张 NVIDIA B200（SM 10.0）、Qwen3.8-27B-NVFP4、PyTorch 2.13.0+cu130、
vLLM 0.31.0 和 FlashInfer 0.7.0.post1。vLLM 实际选择
`FlashInferCuteDslNvFp4LinearKernel`。数据集只有 `rgb + wrist_left` 两路相机；每次 assessment
使用每路 8 帧、0.25 秒间隔、`448×256` JPEG，共 16 张图。视频解码和 JPEG 编码在计时前完成。

| 模式 | 视频 / phase | 视频采样率 | assessments | p50 | p95 | 总推理时间 | 实测持续吞吐 | prefix cache | 票错误 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 单票 | phase 1，10.43 s | 1 Hz | 9 | 293.8 ms | 342.5 ms | 2.629 s | 3.423 Hz | 7.88% | — |
| 五票并发 | phase 1，10.43 s | 1 Hz | 9 | 319.6 ms | 501.8 ms | 3.116 s | 2.888 Hz | 70.89% | 0/45 |
| 五票并发 | phase 1，10.43 s | 2.5 Hz | 22 | 349.6 ms | 409.7 ms | 7.868 s | 2.796 Hz | 59.93% | 1/110 |
| 五票并发 | 未缓存的 phase 2，9.17 s | 5 Hz | 38 | 330.0 ms | 425.6 ms | 13.116 s | **2.897 Hz** | 56.99% | 1/190 |

表中的“视频采样率”决定从视频时间轴取多少个窗口；它不是模型实际完成 assessment 的速度。
benchmark 不 sleep，`实测持续吞吐 = assessments / 总推理时间`。当前五票模式的实测容量上限约为
**2.9 assessments/s**。建议在线先使用 **2.5 Hz**：每个周期 400 ms，而未缓存 phase 2 的
p95 为 425.6 ms，因此单 in-flight 调度偶尔会跳过一个 tick。

五票模式发送五个相同的并发请求，`temperature=0.7`，每票独立 8 秒超时且不重试。错误票弃权，
不会降低门槛；4/5 一致的 terminal 立即接受，3/5 一致则需要下一个新窗口再次确认。相同 prompt
通过 vLLM prefix cache 复用 prefill，所以五票延迟没有变成单票的五倍。

启动五票 adapter：

```bash
PYTHONPATH=src /data/jliu3292/robot-agent-vllm/bin/python \
  -m robot_agent --votes 5 serve --port 8081
```

运行完整视频 benchmark：

```bash
PYTHONPATH=src /data/jliu3292/robot-agent-vllm/bin/python \
  scripts/benchmark_single_vote.py --votes 5 --assessment-hz 2.5 \
  --camera data/g1-pipette-chain-237-241/videos/chunk-000/observation.images.rgb/episode_000237.mp4 \
  --camera data/g1-pipette-chain-237-241/videos/chunk-000/observation.images.wrist_left/episode_000237.mp4 \
  --rotate-180-camera 1 --phase step1_pipette_pick
```

原始 `summary.json` 和逐窗口 `predictions.jsonl` 位于：

- `outputs/benchmark/b200-single-full/`
- `outputs/benchmark/b200-five-full/`
- `outputs/benchmark/b200-five-full-2.5hz/`
- `outputs/benchmark/b200-five-max-phase2/`

完整实验条件、限制和解释见
[docs/b200_nvfp4_five_vote_experiment_report.md](docs/b200_nvfp4_five_vote_experiment_report.md)。当前结果是
两相机离线 steady-state 吞吐测试，还不能替代三相机 live capture-to-verdict 和准确率测试。

这是一个给 Flow Conductor（FC）使用的低延迟视觉 monitor。FC 指定当前 phase，并传入
3 路相机最近 2 秒的窗口；Qwen3.8-27B 返回短判定：

```json
{"status":"in_progress","confidence":0.5}
```

项目不负责机器人控制、phase 排序或重规划。它只验证当前 phase 的可见物理结果，并保存已经
确认的 phase 状态。

## 线上参数

- 支持确定性单票（`temperature=0`）和五票并发（`temperature=0.7`）。
- 3 路相机 × 8 帧，间隔 0.25 秒，共 24 张 `448×256` JPEG。
- 关闭 thinking，`max_tokens=32`，请求超时 8 秒，不重试。
- assessment rate 可配置；当前 B200 五票建议从 2.5 Hz 开始。单票需要两个新窗口确认；五票按
  4/5 立即接受、3/5 下一窗口确认。
- vLLM 开 prefix caching；system、phase rubric 和帧顺序说明位于变化图像之前。
- backend 超时或格式错误按 `in_progress/confidence=0` 返回，并单独暴露错误类型；不能伪装成
  视觉失败。

所有参数集中在 [configs/monitor.toml](configs/monitor.toml)。架构细节见
[docs/architecture.md](docs/architecture.md)。

## A5000 启动

RTX A5000（SM 8.6）不能运行当前 NVFP4 checkpoint，使用 4-bit AWQ/Marlin 双卡 TP=2。
先把完整 snapshot 下载到本地，之后服务过程不再依赖 Hugging Face：

```bash
bash scripts/download_qwen_awq.sh /data/jingyu/Qwen3.8-27B-AWQ

CUDA_VISIBLE_DEVICES=0,1 \
MODEL_ENV=/data/jingyu/H-PEEK/.venv \
MODEL_PATH=/data/jingyu/Qwen3.8-27B-AWQ \
bash scripts/serve_qwen_awq.sh
```

下载/启动脚本会补齐该 checkpoint 缺失的 FP16 模块排除元数据，并给 vLLM 0.19.x 应用一条
Qwen3.5 视觉权重前缀映射；重复执行是幂等的。

默认使用 AWQ-Marlin、FP16 activation、8192 token 上下文、16 张图上限和 prefix caching；
视觉 encoder 在每张卡上保留完整权重并按输入做 data parallel，避免量化视觉层被 TP 切坏。
当前双 A5000 路径使用 NCCL all-reduce 和 eager execution，绕过该拓扑下 CUDA graph custom
all-reduce 捕获的 `invalid argument`。
NVFP4 的诊断/启动脚本仍保留，供 compute capability ≥ 8.9 的机器使用。

另开终端启动 FC HTTP adapter：

```bash
PYTHONPATH=src python -m robot_agent serve
```

接口：

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/observe \
  -H 'Content-Type: application/json' --data-binary @examples/observation.json
curl -X POST http://127.0.0.1:8080/reset \
  -H 'Content-Type: application/json' -d '{}'
```

每个进程只维护一个 episode。输入必须包含 `episode_id`、严格递增的 `frame_id/timestamp_s`、
`expected_phase` 和恰好 24 张 camera-major 图像。episode 结束时传 `end_of_episode=true`；若五个
phase 未全部确认，返回 `INCOMPLETE_AT_END`。

## 单票频率测试

必须使用真实、不断变化的视频，不能用重复合成图制造 MM cache 命中。生产目标是三路相机；
当前 G1 数据集只有 `rgb + wrist_left` 两路，因此报告会明确记录 `camera_count=2`：

```bash
PYTHONPATH=src /path/to/model-env/bin/python scripts/benchmark_single_vote.py \
  --camera data/g1-pipette-chain-237-241/videos/chunk-000/observation.images.rgb/episode_000237.mp4 \
  --camera data/g1-pipette-chain-237-241/videos/chunk-000/observation.images.wrist_left/episode_000237.mp4 \
  --rotate-180-camera 1 \
  --phase step1_pipette_pick
```

脚本先顺序解码并缓存所需帧，视频准备时间不计入模型推理；随后以 1 Hz 遍历整个第一阶段，
将每次原始回复、解析结果、agent 状态、token/cache 和延迟写到 `predictions.jsonl`，汇总完整
响应的 p50/p95 和串行 Hz 到 `summary.json`。本机 A5000 的实测阻塞原因见
[docs/single_vote_benchmark.md](docs/single_vote_benchmark.md)。

下载同一原始 chain 切出的五个相邻 phase：

```bash
HF_BIN=/path/to/hf bash scripts/download_g1_chain.sh
```

在本机 A5000 上一键启动 AWQ 服务、用不同视频预热并测试第一阶段：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_ENV=/data/jingyu/H-PEEK/.venv \
MODEL_PATH=/data/jingyu/Qwen3.8-27B-AWQ \
bash scripts/run_phase1_awq.sh
```

默认测试结束后只关闭本脚本启动的服务；设置 `KEEP_SERVER=1` 可保留服务。vLLM 日志、预热
记录、前后 metrics、phase 1 predictions 和 summary 都保存在
`outputs/benchmark/phase1-awq/`。

本机双 A5000 实测 9/9 请求有效，完整响应 p50 `2063.9 ms`、p95 `2156.6 ms`，串行单票
`0.485 Hz`；详见 [docs/single_vote_benchmark.md](docs/single_vote_benchmark.md)。

## 单卡 7B 基线

官方 `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` 可在单张 RTX A5000 上运行：

```bash
bash scripts/download_qwen25_vl_7b_awq.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase1_qwen25_vl_7b_awq.sh
```

同一 phase 1 输入实测 p50 `819.3 ms`、p95 `1011.0 ms`，串行单票 `1.221 Hz`，9/9
请求有效。它在 9.75 秒才首次判定 success，因视频结束前没有第二个新窗口而未接受终态。
与 27B 的延迟和判断对比见
[docs/single_gpu_qwen25_vl_7b_benchmark.md](docs/single_gpu_qwen25_vl_7b_benchmark.md)。

## 开发验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m robot_agent --backend mock replay
```

模型权重、视频、输出和虚拟环境不应提交到仓库。
