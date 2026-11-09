"""Normalize Qwen3.8 AWQ metadata for vLLM's mixed FP16/AWQ loader."""

import argparse
import json
from pathlib import Path


REQUIRED_UNQUANTIZED = ("visual", "lm_head", "in_proj_a", "in_proj_b")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    path = args.model / "config.json"
    config = json.loads(path.read_text())
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "awq" or quant.get("bits") != 4:
        raise SystemExit("checkpoint is not 4-bit AWQ")
    current = list(quant.get("modules_to_not_convert") or [])
    updated = current + [name for name in REQUIRED_UNQUANTIZED if name not in current]
    if updated != current:
        quant["modules_to_not_convert"] = updated
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"updated {path}: modules_to_not_convert={updated}")
    else:
        print(f"checkpoint metadata already prepared: {path}")


if __name__ == "__main__":
    main()
