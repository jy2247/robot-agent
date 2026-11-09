# NVFP4 自托管

权重默认位于 `/data/jingyu/Qwen3.8-27B-NVFP4`。下载支持续传：

```bash
bash scripts/download_qwen_nvfp4.sh /data/jingyu/Qwen3.8-27B-NVFP4
```

启动前先独立检查 checkpoint 和 GPU：

```bash
MODEL_ENV=/path/to/model-env \
/path/to/model-env/bin/python scripts/check_nvfp4_runtime.py \
  /data/jingyu/Qwen3.8-27B-NVFP4
```

ModelOpt NVFP4 需要 compute capability ≥ 8.9。检查通过后：

```bash
MODEL_ENV=/path/to/model-env \
MODEL_PATH=/data/jingyu/Qwen3.8-27B-NVFP4 \
bash scripts/serve_qwen_nvfp4.sh
```

服务参数与延迟设计文档一致：24576 context、20480 batched tokens、8 sequences、24 images、
FP8 KV cache、prefix caching、0.85 GPU memory utilization，以及 `[1,2,4,8]` CUDA graph capture。
实际部署必须预热后用真实三路视频运行 `scripts/benchmark_single_vote.py`。

对于已下载的 G1 双视角 chain，可以直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_ENV=/path/to/model-env \
MODEL_PATH=/path/to/Qwen3.8-27B-NVFP4 \
bash scripts/run_phase1_nvfp4.sh
```

脚本会用 episode 241 预热 kernel/视觉/decode 路径，再测 episode 237，避免 phase 1 的图片
直接命中预热时的 multimodal cache。
