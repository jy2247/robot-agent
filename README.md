# Robot VLM Monitor

这是一个给 Flow Conductor（FC）使用的低延迟视觉 monitor。FC 指定当前 phase，并传入
3 路相机最近 2 秒的窗口；Qwen3.8-27B-AWQ 只返回一次短判定：

```json
{"status":"in_progress","confidence":0.5}
```

项目不负责机器人控制、phase 排序或重规划。它只验证当前 phase 的可见物理结果，并保存已经
确认的 phase 状态。

## 线上参数

- 单票，不使用 5 票并发；`temperature=0`。
- 3 路相机 × 8 帧，间隔 0.25 秒，共 24 张 `448×256` JPEG。
- 关闭 thinking，`max_tokens=32`，请求超时 8 秒，不重试。
- 每秒一次 assessment，同一 terminal 结果必须在两个新窗口中连续出现才接受。
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
