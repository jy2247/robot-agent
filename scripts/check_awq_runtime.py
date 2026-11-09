"""Validate a local AWQ checkpoint and the visible CUDA devices."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import torch


def package_version(name: str):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.model / "config.json"
    index_path = args.model / "model.safetensors.index.json"
    problems = []
    config = {}
    if not config_path.is_file() or not index_path.is_file():
        problems.append("checkpoint metadata is incomplete")
    else:
        config = json.loads(config_path.read_text())
        quant = config.get("quantization_config", {})
        if quant.get("quant_method") != "awq" or quant.get("bits") != 4:
            problems.append("checkpoint is not 4-bit AWQ")
        required = {"visual", "lm_head", "in_proj_a", "in_proj_b"}
        skipped = set(quant.get("modules_to_not_convert") or [])
        if not required.issubset(skipped):
            problems.append("AWQ metadata does not exclude all FP16 modules")

    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            gpus.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": f"{capability[0]}.{capability[1]}",
                    "memory_free_gib": round(free_bytes / 2**30, 2),
                    "memory_total_gib": round(total_bytes / 2**30, 2),
                    "awq_marlin_capability_ok": capability >= (8, 0),
                }
            )
            if capability < (8, 0):
                problems.append(f"GPU {index} does not support the Ampere AWQ/Marlin path")
    else:
        problems.append("CUDA is unavailable")

    if len(gpus) < args.tensor_parallel_size:
        problems.append(
            f"tensor parallel size {args.tensor_parallel_size} requires more visible GPUs"
        )

    report = {
        "status": "ready" if not problems else "blocked",
        "checkpoint": str(args.model),
        "quantization": config.get("quantization_config"),
        "architecture": config.get("architectures"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": package_version("vllm"),
        },
        "gpus": gpus,
        "problems": problems,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
