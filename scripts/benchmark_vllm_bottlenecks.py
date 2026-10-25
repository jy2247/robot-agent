#!/usr/bin/env python3
"""Measure client, vision/prefill, decode, and batching costs against vLLM."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import statistics
import sys
import time
from urllib.request import Request, urlopen

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robot_agent.config import load_config
from robot_agent.context import messages


PROMPT = (
    "Inspect the ordered robot-camera images. Reply with one compact JSON object "
    'only: {"phase":"unknown","status":"executing","confidence":0.5}. '
    "Do not explain."
)


def read_frames(head_path: Path, wrist_path: Path):
    by_view = []
    for path in (head_path, wrist_path):
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 52:
            raise RuntimeError(f"not enough frames in {path}")
        frames = []
        for index in (round((count - 1) * step / 51) for step in range(52)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode {path}:{index}")
            frames.append(frame)
        capture.release()
        by_view.append(frames)
    # Alternate synchronized head/wrist views. Every request below consumes
    # fresh frames so multimodal-cache hits cannot hide vision-tower cost.
    return [frame for pair in zip(*by_view) for frame in pair]


def encode_images(frames, side):
    started = time.perf_counter()
    urls = []
    byte_count = 0
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((side, side))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=82)
        encoded = output.getvalue()
        byte_count += len(encoded)
        urls.append("data:image/jpeg;base64," + base64.b64encode(encoded).decode())
    return urls, (time.perf_counter() - started) * 1000, byte_count


def payload(model, images, max_tokens, stream):
    content = [{"type": "text", "text": PROMPT}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
    }


def request_once(base_url, body, timeout):
    body = {key: value for key, value in body.items() if value is not None}
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_s = None
    usage = {}
    text_chunks = []
    with urlopen(request, timeout=timeout) as response:
        if body["stream"]:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices", ())
                if choices:
                    content = choices[0].get("delta", {}).get("content")
                    if content:
                        if first_token_s is None:
                            first_token_s = time.perf_counter() - started
                        text_chunks.append(content)
        else:
            result = json.load(response)
            usage = result.get("usage", {})
            text_chunks.append(result["choices"][0]["message"]["content"])
    elapsed_s = time.perf_counter() - started
    return {
        "latency_ms": round(elapsed_s * 1000, 1),
        "ttft_ms": round(first_token_s * 1000, 1) if first_token_s is not None else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "output_chars": len("".join(text_chunks)),
    }


def summarize(samples):
    result = {"samples": samples}
    for key in ("latency_ms", "ttft_ms"):
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        if values:
            result[f"median_{key}"] = round(statistics.median(values), 1)
            result[f"min_{key}"] = round(min(values), 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--head", type=Path,
                        default=Path("outputs/continuous-source-cache/g1/episode_000237/head.mp4"))
    parser.add_argument("--wrist", type=Path,
                        default=Path("outputs/continuous-source-cache/g1/episode_000237/left_wrist.mp4"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--agent-config", default="configs/g1_pipette_fast.toml")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frames = read_frames(args.head, args.wrist)
    encoded = {}
    encoding = {}
    for side in (224, 336, 448):
        encoded[side], encode_ms, byte_count = encode_images(frames, side)
        encoding[side] = {"all_image_encode_ms": round(encode_ms, 1),
                          "mean_image_jpeg_bytes": round(byte_count / len(frames), 1)}

    scenarios = (
        ("text_only", 336, 0),
        ("one_image_336", 336, 1),
        ("two_images_336", 336, 2),
        ("four_images_224", 224, 4),
        ("four_images_336", 336, 4),
        ("four_images_448", 448, 4),
    )
    results = {"encoding": encoding, "scenarios": {}, "concurrency": {}}
    cursor = 0
    for name, side, count in scenarios:
        # Warm the compiled model while keeping each visual input unique.
        body = payload(args.model, encoded[side][cursor:cursor + count], args.max_tokens, True)
        request_once(args.base_url, body, args.timeout)
        cursor += count
        samples = []
        for _ in range(args.repeats):
            body = payload(args.model, encoded[side][cursor:cursor + count],
                           args.max_tokens, True)
            samples.append(request_once(args.base_url, body, args.timeout))
            cursor += count
        results["scenarios"][name] = summarize(samples)
        print(json.dumps({name: results["scenarios"][name]}), flush=True)

    for concurrency in (1, 2, 4):
        bodies = []
        for _ in range(concurrency):
            bodies.append(payload(args.model, encoded[336][cursor:cursor + 4],
                                  args.max_tokens, True))
            cursor += 4
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            samples = list(executor.map(
                lambda body: request_once(args.base_url, body, args.timeout), bodies))
        elapsed_s = time.perf_counter() - started
        results["concurrency"][str(concurrency)] = {
            "wall_ms": round(elapsed_s * 1000, 1),
            "requests_per_s": round(concurrency / elapsed_s, 3),
            **summarize(samples),
        }
        print(json.dumps({f"concurrency_{concurrency}":
                          results["concurrency"][str(concurrency)]}), flush=True)

    config = load_config(args.agent_config)
    samples = []
    for sample_index in range(args.repeats):
        urls = encoded[336][cursor:cursor + 4]
        cursor += 4
        history = [{"episode_id": "throughput-profile", "frame_id": f"h{sample_index}",
                    "timestamp_s": sample_index * 2.0, "expected_phase": config.phases[0],
                    "note": "Ordered simultaneous camera views: head, then left_wrist.",
                    "images": urls[:2]}]
        current = {"episode_id": "throughput-profile", "frame_id": f"c{sample_index}",
                   "timestamp_s": sample_index * 2.0 + 1.0,
                   "expected_phase": config.phases[0],
                   "note": "Ordered simultaneous camera views: head, then left_wrist.",
                   "images": urls[2:]}
        body = {"model": config.model,
                "messages": messages(config, history, current, {}),
                "temperature": 0, "max_tokens": config.max_tokens, "stream": True,
                "stream_options": {"include_usage": True}, **config.model_extra_body}
        if config.chat_template_kwargs:
            body["chat_template_kwargs"] = config.chat_template_kwargs
        samples.append(request_once(args.base_url, body, args.timeout))
    results["actual_agent_prompt"] = summarize(samples)
    print(json.dumps({"actual_agent_prompt": results["actual_agent_prompt"]}), flush=True)

    rendered = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
