"""Add the missing Qwen3.5 AWQ vision-prefix mapping to vLLM 0.19.x."""

from pathlib import Path

import vllm


def main() -> None:
    path = Path(vllm.__file__).parent / "model_executor/models/qwen3_vl.py"
    source = path.read_text()
    rule = '            "model.language_model.visual.": "visual.",\n'
    if rule in source:
        print(f"vLLM Qwen3.5 vision mapping already present: {path}")
        return
    anchor = (
        "        orig_to_new_prefix={\n"
        '            "model.visual.": "visual.",\n'
        '            "lm_head.": "language_model.lm_head.",\n'
        '            "model.language_model.": "language_model.model.",\n'
    )
    if source.count(anchor) != 1:
        raise SystemExit(f"unsupported vLLM Qwen3 VL mapper layout: {path}")
    replacement = (
        "        orig_to_new_prefix={\n"
        "            # Native-AWQ Qwen3.5 exports may nest vision here.\n"
        + rule
        + '            "model.visual.": "visual.",\n'
        + '            "lm_head.": "language_model.lm_head.",\n'
        + '            "model.language_model.": "language_model.model.",\n'
    )
    path.write_text(source.replace(anchor, replacement))
    print(f"patched vLLM Qwen3.5 vision mapping: {path}")


if __name__ == "__main__":
    main()
