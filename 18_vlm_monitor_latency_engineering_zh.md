# 27B 多模态模型做实时机器人 monitor：延迟优化的设计决策

目标：用本地 vLLM 跑 Qwen3 27B（视觉语言模型），每秒对 3 路相机的最近 2 秒画面判定一次"当前 skill 是 成功 / 进行中 / 失败"，并把延迟压到机器人可以用的范围。
硬件：1 张 RTX 5090（32 GB）。
下面按**对延迟贡献从大到小**列出决策，每条给出改前/改后的实测数字和原因；最后附完整的 server 启动参数和请求示例，可以直接照抄。

## 0. 先对齐"延迟"指什么

同一套系统有三个不同的延迟，不能混着说：

| 指标 | 含义 | 我们达到的 |
|---|---|---|
| TTFT | prefill 完、吐出第一个 token | 24 张 256² 图：**175-195 ms**；24 张 448² 图：**~500 ms** |
| 单票判定 | 一个请求完整返回 `{"status","confidence"}`（约 16 token） | **0.33 s**（预热、合成图）/ **0.5 s**（真实回放，p95 0.8 s） |
| 一次 assessment（5 票投票） | 5 个采样全部返回并投票 | **0.9-1.15 s** 中位，p95 1.3-1.4 s |
| 线上端到端 | 相机 snapshot → 判定被接受 | **1.17 s** 中位，p95 1.35 s，最大 2.25 s（756 次） |

"100-200 ms"只是 TTFT。完整判定在这个架构下是 **1 s 量级**，我们的做法是 1 s 节拍、机器人不停、判定 1.2 s 内回来。

### 延迟模型（5090 + NVFP4 + 不思考）

- prefill ≈ **0.1 ms × prompt token 数 + 40-70 ms 固定开销**
- decode ≈ **67 token/s**，每个输出 token 15 ms
- 图片 token 数 = (宽/32) × (高/32)（Qwen3-VL：patch 16，2×2 merge）
- 5 票并发时 2-5 票命中 prefix cache，几乎只付一次 prefill

用这个公式可以先估再测。例：24 张 448×256 → 112 token/张 → 2.7k 图片 token + ~1.5k 文字 ≈ 4.2k → prefill ≈ 0.45 s，decode 16 token ≈ 0.25 s，单票 ≈ 0.7 s 上限，实测 0.5 s。

## 1. 决策清单（按贡献排序）

### D1. 关掉 thinking —— 单票 2.5 s → 0.33 s，这一条占了绝大部分收益

Qwen3 默认先输出一段 `<think>` 再答，实测每次 20-30 s 级别的思考，160+ 个 token，60% 的请求撞到 2,048 token 上限还没给出答案，还会触发客户端的格式校验重试（一次判定变成两个请求）。

做法：每个请求带
```json
"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}
```
效果：同一输入，中位 2.54 s / 163 输出 token → **0.33 s / 16 token**。
注意：server 端的 `--reasoning-parser qwen3` 只负责解析 think 标签，**不会**让模型不思考；必须在请求里关。
附带发现：thinking 开不开对 TTFT 没影响，它只花在 decode 上，所以这一条省的全是 decode 时间。

### D2. 输出只要两个字段的 JSON，不用 guided decoding —— 决定 decode 时长

decode 是平坦的 67 token/s，输出每多 10 个 token 多 150 ms，所以输出长度直接决定延迟。

做法：
- 输出 schema 固定为 `{"status": "in_progress|success|fail", "confidence": 0.0-1.0}`，约 16 token
- 格式靠 prompt 末尾一句示例保证："Output one JSON object, for example {"status":"in_progress","confidence":0.5}, without explanation or a JSON Schema"
- **不用** `response_format` / JSON schema 约束解码（有额外开销，而且 Qwen 对示例的服从度已经够）
- 客户端解析回复里的**第一个**合法 JSON 对象，容忍代码围栏和多余文字
- `max_tokens` 设 24-128 做兜底

试过的反例：
- 想用更"紧凑"的输出格式（只回一个状态词）反而 0.93 s——模型多说了 57 个 token
- 加一句 short reason：0.41 → 0.54 s（+33%），输出 21 → 30 token。调试时开，部署时关

### D3. 本地部署 + NVFP4 混合精度，让 27B 装进 32 GB

- 同一模型走云 API（OpenRouter）首 token p50 **2,477 ms** vs 本地 **193 ms**，排队和网络抖动完全不可控。本地是前提。
- 用 NVFP4 的 checkpoint（compressed-tensors）：MLP 为 W4A4（group 16），attention / 线性投影 / lm_head / 最后 8 层 MLP 为 FP8，KV cache FP8，vision tower BF16。磁盘 23 GB，加载后 21 GB 显存，静态占用 26.7 GB。
- vLLM 0.29，FP4 GEMM 走 FlashInfer CUTLASS kernel（sm120 验证过）。
- 同卡试过 SGLang：TTFT 147 vs 99 ms，decode 40 vs 67 tok/s，慢 1.4-2×，放弃。

### D4. 控制图片的 token：分辨率 ~448×256、3 路 × 8 帧 —— prefill 的主项

prompt token 里图片占 2/3 以上，文字（rubric、规则）只占 1-1.5k，所以 prefill 时间基本由"几张图 × 每张多少 token"决定。

实测（24 张，batch 1）：

| 每张分辨率 | token/张 | 24 张 prompt | TTFT |
|---|---|---|---|
| 256×256 | 64 | 1.6k | 175-195 ms |
| 448×256（我们线上） | 112 | 2.7k + 文字 ≈ 4.7-6k | ~500 ms |
| 448×448 | 196 | 4.8k | ~500 ms |
| 896×896 | 784 | 18.9k | 2.4-3 s |

我们的选择：相机原生 424×240，宽度不放大，右/下 pad 到 32 的倍数 → 448×256；JPEG quality 95、不做色度子采样；每路相机取最近 8 帧、间隔 0.25 s、最老不超过 2 s；共 24 张。
原因：8 帧 × 0.25 s 是准确率实验里需要的时间上下文；分辨率再往上准确率没涨、延迟线性涨。
新图片的 CPU 预处理每张 1-3 ms，可以忽略。p99 − p50 只有 1-10 ms，尾部很紧，所以 p50 就够用来做预算。

### D5. 同一相机的 8 帧打包成"视频"而不是 8 张图 —— token 减半，5 票延迟 −16%

Qwen3-VL 的视频处理器把相邻两帧做 temporal patch 合并，同样 24 帧：图片模式 2,688 token，视频模式 **1,344 token**。
做法：每路相机一段 `video_url`，内容是 `data:video/jpeg;base64,<帧1>,<帧2>,...`，并在 `extra_body` 里给出每帧的时间戳（`media_io_kwargs.video.frames_indices`，单位 ms）和 `do_sample_frames: false`、`mm_processor_kwargs.do_resize: false`，让 server 不再重采样/缩放。
效果：真实回放 143 个 snapshot × 3 次，5 票中位 1.056 → **0.890 s**，p95 1.43 → 1.32 s；准确率不变（84.4% vs 85.1%）。

### D6. 开 prefix caching，并把 prompt 按"静态在前、变化在后"排 —— 5 票几乎只付一次 prefill

vLLM 的 prefix cache 按 token 前缀匹配，第一个不同的 token 之后全部重算。

做法：
- server 开 `--enable-prefix-caching`
- prompt 顺序：system（判定规则 + 输出 schema + 该 skill 的 rubric，全部固定）→ user 文字（任务上下文）→ 每帧的元信息 → 图片/视频
- 一次 assessment 的 5 票 prompt **逐字节相同**，所以第 2-5 票的 prefill 全部命中，这是"5 票只比 1 票慢 2 倍而不是 5 倍"的原因
- 跨 tick 能复用的只有静态文字部分（图片每 tick 都变）；日志里全局命中率 46-60%

还能再压的：我们把每帧"received 0.xxx s before this snapshot"这种每次都变的字符串放在图片前面，跨 tick 的可复用前缀到此为止；把它挪到图片之后或量化到 0.25 s 网格会更好，没来得及做。

### D7. 5 票采样怎么发 —— 并发 5 个请求，各自超时，失败弃权

投票是为了对冲 temperature 0.7 下的抖动（同一输入两次判定不一致的比例约 26%）。

做法（线上）：5 个并发 HTTP 请求（`asyncio.gather`），每票独立 8 s 超时；超时或出错的票**弃权**但不降低阈值；客户端 `retries: 0`。
另一种（离线 benchmark）：一个请求 `n=5`，5 个 choice 分别解析，HTTP 开销小一点，但一个请求失败就全没了。
两种延迟差不多（都吃 prefix cache），线上选并发是为了单票失败不拖累整次。
观察：`--max-num-seqs 8` 下引擎同时 Running 最多 5-6 个请求，**没有打满**，瓶颈在单请求的 prefill + decode，不在吞吐。

### D8. 调度：1 s 节拍、同时只允许一个 assessment 在飞、晚到就跳过

- 每 1.0 s 发起一次 assessment，机器人运动中也判
- 上一次没回来就不发新的（也不排队），直接按"进行中"处理 → 实际节拍 = max(1 s, 延迟) ≈ 1.2 s
- judge 调用失败/超时一律当"进行中"，不当判定；每票 8 s 超时；skill 自身有 30 s 预算
- 只用请求发出之后到达的帧做证据，避免用到陈旧画面
- 接受规则：5 票里 4 或 5 票一致的 terminal（成功/失败）立即接受；恰好 3 票要**下一次** assessment（有新 snapshot、至少一路相机有新帧）同样 3 票才接受；多数是"进行中"或没有多数就清空
- 结果：弱信号最多 2 个节拍，强信号 1 个节拍；判定延迟（相对真值事件）中位 0.75 s，p95 1.75 s

典型时间线：snapshot 8.624 s → 5 票返回 10.041 s → 判定接受 10.044 s。

### D9. vLLM 显存和批参数 —— 全是踩 OOM 踩出来的

| 参数 | 值 | 原因 |
|---|---|---|
| `--gpu-memory-utilization` | 0.85 | 0.90 在大图输入时运行期 OOM；0.93 在 CUDA graph 捕获期 OOM |
| `--compilation-config cudagraph_capture_sizes` | `[1,2,4,8]` | 默认列表太长导致捕获期 OOM；并发最多 5-6，8 够 |
| `--max-num-batched-tokens` | 20480 | vLLM 的 encoder cache 要一次装下**一个请求的全部图片 token**，这个值必须 ≥ 单请求图片 token 总数，否则请求被拒 |
| `--max-model-len` | 24576 | 同上留余量；KV 每 token 32 KiB（FP8），开 prefix cache 后可用 KV 约 28k token |
| `--max-num-seqs` | 8 | 5 票并发 + 余量 |
| `--limit-mm-per-prompt` | `{"image": 24}` | 3 × 8 |
| `--mm-processor-kwargs` | `{"min_pixels":65536,"max_pixels":802816}` | 每张图 token 上限 784 |
| 环境变量 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 减少碎片 OOM |

引擎冷启动 110 s（其中编译 34 s），要预热后再压测。

### D10. 其他小决定

- 采样 `temperature 0.7, top_p 1.0`。注意 vLLM 会用模型自带的 `generation_config.json` 覆盖默认值（Qwen3 是 `top_k 20, top_p 0.95`），不传 `--generation-config vllm` 的话 top_k=20 会一直在；我们没覆盖它
- 不 stream（延迟就是整响应 wall time）；不用 logprobs（要的话每个 token 都多算一份）
- Qwen 的 chat template 只允许**一个** system message 且必须在最前，否则 400 "System message must be at the beginning"；客户端把多个 system 合并成一个
- 压测**不要**用重复的合成图片：会全部命中 prefix/MM cache，数字偏乐观一倍以上。上面的 TTFT 表是关掉 prefix cache 测的，线上数字是真实回放测的

## 2. 试过但没用或反效果的

| 尝试 | 结果 |
|---|---|
| 更"紧凑"的输出 schema（只回状态词） | 0.93 s vs 0.33 s，模型多吐 token |
| 加 short reason 输出 | +33% 延迟，准确率没涨；只在调试模式用 |
| 896² 高分辨率 | TTFT 2.4-3 s，且 24 张直接 OOM（gpu-memory-utilization 0.90） |
| SGLang | 比 vLLM 慢 1.4-2× |
| 云 API | 首 token 2.5 s |
| 帧间隔 0.25 → 0.5 → 1.0 s（同样 24 帧） | 5 票延迟 1.337 / 1.330 / 1.327 s，几乎一样——延迟由帧**数**决定，不由时间跨度决定 |

## 3. 还能再压的（我们没做）

1. 把每帧时间戳字符串挪到图片后面或量化，延长跨 tick 的 prefix 命中
2. 线上全面切视频模式（D5），目前只在回放里验证过
3. 24 帧 → 16/12 帧：TTFT 线性下降，需要重新验证准确率
4. speculative decoding：checkpoint 带 MTP 头但我们没加载，16 token 的 decode 有空间
5. 用 vLLM 的 Prometheus 直方图（`vllm:time_to_first_token_seconds`、`request_prefill_time_seconds`）把 1.0 s 拆成 prefill / decode / HTTP，我们只量了整响应
6. 节拍 1 s 但延迟 1.17 s，每次都跳一拍：节拍放到 1.25 s，或允许 2 个 in-flight

## 4. 可以直接照抄的配置

### server

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm/vllm-openai:latest \
  RedHatAI/Qwen3.8-27B-NVFP4 \
  --max-model-len 24576 \
  --max-num-batched-tokens 20480 \
  --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image":24}' \
  --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":802816}' \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'
```

### 请求（OpenAI 兼容 `/v1/chat/completions`，图片模式）

```json
{
  "model": "RedHatAI/Qwen3.8-27B-NVFP4",
  "messages": [
    {"role": "system", "content": "<判定规则 + 输出 schema 说明 + 本 skill 的 rubric，全部固定>"},
    {"role": "user", "content": [
      {"type": "text", "text": "<任务上下文>\n<每帧的相机名和时间偏移>"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      "... 共 24 张，按相机分组、时间升序 ...",
      {"type": "text", "text": "Output one JSON object, for example {\"status\":\"in_progress\",\"confidence\":0.5}, without explanation or a JSON Schema."}
    ]}
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 1.0,
  "n": 5,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

线上版本把 `n: 5` 换成 5 个并发请求，每个 8 s 超时；图片换成每路相机一个 `video_url`（见 D5）。

### 建议的验证顺序

1. `max_tokens=1`、关 prefix cache、真实图片，量 TTFT，对齐 D4 的表
2. 开 prefix cache，量单票完整返回
3. 量 5 票并发
4. 接上相机，量 snapshot → 判定接受的端到端
