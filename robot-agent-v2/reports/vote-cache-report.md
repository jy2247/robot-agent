# RTX 5080：3/5 vote 与 KV prefix cache 实测

模型：Qwen/Qwen3-VL-8B-Instruct-FP8。相同真实双相机录像 237–241，50 个窗口，每路 8 帧、间隔 0.25 秒、448×256。各次运行均记录原始逐票输出，不用重复合成图像。

## 核心结果

| 模式 | 有效窗口 | Hz | p50 / p95 ms | accepted 阶段 | 安全停止窗口 |
|---|---:|---:|---:|---:|---:|
| 1 vote | 50/50 | 1.071 | 930.4 / 962.9 | 2/5 | 6 |
| 3 votes | 50/50 | 0.389 | 2552.7 / 2649.2 | 2/5 | 6 |
| 5 votes | 50/50 | 0.241 | 4120.8 / 4242.2 | 2/5 | 6 |
| 3 votes + KV | 50/50 | 0.582 | 1698.6 / 1793.0 | 2/5 | 6 |
| Contact-rule safety | 50/50 | 1.047 | 949.1 / 979.3 | 2/5 | 7 |

3 vote 开启 KV cache 后，吞吐提升 49.6%，p50 延迟降低 33.5%。这些是整组双 critic、多票决策的吞吐和延迟，不是单 token 速度。

## 缓存复用了什么

视觉 encoder 结果在同一窗口的两个 critic 和全部投票间共享。KV prefix cache 进一步复用每个 critic 自己的相同前缀；两个 critic 的系统提示词不同，通常不能共用整份语言 KV。每个新窗口使用新的图像 UUID，避免把旧画面当成新证据；匹配的纯文本系统前缀仍可复用。缓存不复用最终答案，各票仍分别采样、解码。
本次 3-vote KV 运行共命中 593,232 / 821,310 个 prompt token（72.2%）。命中数与视觉缓存的 image 命中数分开记录。

温度 0.7、top-p 0.9，每票使用固定 seed 0、1、2（5 vote 增加 3、4），每个模型响应仍是两个字段、最多 32 tokens；整组 8 秒截止，不重试。三次重复确定性生成不会创造新的视觉证据。

## 为什么中间三个阶段没有 accepted

- 阶段 2：原 1-vote 运行全部为 in_progress。confidence=0.8 是对该状态的置信度，不是完成概率。
- 阶段 3：20.75 和 26.75 秒分别 success，但不连续；其余窗口大多 in_progress。保持连续两个窗口确认时，增加票数也没有解决。
- 阶段 4：全部为 in_progress，没有充分确认释放后稳定支撑。安全为 safe 的窗口也没有阶段成功，因此不是单纯被安全 critic 拦截。
- 3/5 vote 都没有新增稳定 accepted 的阶段。多数决要求配置票数的严格多数且各 terminal 票 confidence≥0.70；非法/低置信票不能缩小分母。终态 confidence 是获胜票均值，不人为升高。没有终态多数时，聚合器以 in_progress/confidence=0 表示弃权，原始 confidence 保留在逐票记录中。

额外对相同 5-vote 输出应用仓库已有的“≥4/5 票立即接受”规则，可在 20.75 秒接受阶段 3；阶段 2、4 仍不通过。该离线政策比较没有重新推理，不是准确率提升，也没有单独测量 Hz。默认连续确认规则没有被放宽。详见 `rtx5080-vote5/fast-consensus-policy.json`。

## 安全 verifier 独立测试

新增 support_contact 提示词，强调手掌/手指承托属于 hand、投影重叠不能证明桌面承托、支撑接触无法判断时输出 unknown。只改变安全提示，阶段提示、阈值及连续确认规则不变。
完整测试没有修复关键管子支撑误判；安全停止窗口从 6 增至 7，因此该候选没有被部署。输出 schema 约束只能防止非法枚举，不能修复物体识别。安全分歧聚合成 unknown，不会被伪装成 safe；真实 unsafe 多数仍保留。
当前仍需独立标注的持握/支撑证据，以及真实掉落、错误放置等负例，才能评价和改进安全准确率。单一成功源录像上的停止窗口数不等于已标注误报率。

## 缓存前后输出一致性

```json
{
  "phase_status_matches": 50,
  "phase_accept_matches": 50,
  "safety_support_matches": 50,
  "individual_raw_matches": 288,
  "individual_raw_total": 300,
  "windows": 50
}
```

## 测量边界与复现

最终 KV 测试没有与视频导出任务并行。前一次探索测试的早期窗口与 CPU 视频导出重叠，保存在 outputs/rtx5080-vote3-kv-exploratory，未用于最终比较。模型加载、warmup、视频/JPEG 准备不计入 Hz；每个窗口的图像处理、prefill、全部 decode 和聚合计入整组时延。未测量真实摄像头到控制器的端到端延迟。
真实阶段完成时刻尚未独立精确标注，录像切段不是 ground truth；因此没有宣称已知真实转换检测延迟。该实验不是三相机生产验证，也未连接机器人。

```bash
robot-agent-v2/.venv-vllm/bin/python robot-agent-v2/benchmark.py --votes 3 --prefix-cache --schedule recorded --output robot-agent-v2/outputs/rtx5080-vote3-kv --annotations robot-agent-v2/annotations/chain-237-241.json
```

本地 API 可用 `serve.py --cameras 2 --votes 3 --prefix-cache --port 8082` 启动。代码默认仍为 1 vote；3/5 vote 是用户显式启用的多请求实验模式。当前实验服务使用 3 vote + KV，以便检查缓存行为；它未解决中间阶段和安全误判问题。

36 项单元测试（原项目 18 + V2 18）、mock replay、API 真实窗口/缓存命中/重复帧拒绝/reset，以及浏览器回放检查均记录在 reports 中。

[完整对照 JSON](vote-cache-comparison.json) · [逐票 KV 结果](rtx5080-vote3-kv/predictions.jsonl) · [缓存运行 summary](rtx5080-vote3-kv/summary.json) · [可视化](http://127.0.0.1:8765/experiments/)
