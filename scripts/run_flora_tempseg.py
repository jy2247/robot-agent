#!/usr/bin/env python3
"""Run NVIDIA Flora-TempSeg only, without loading Flora-Captioner."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def _python310_file_digest_compat() -> None:
    if hasattr(hashlib, "file_digest"):
        return

    def file_digest(stream, digest):
        hasher = digest() if callable(digest) else hashlib.new(digest)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        return hasher

    hashlib.file_digest = file_digest  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Flora temporal segmenter on one video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--flora-repo",
        type=Path,
        default=Path(os.environ.get("FLORA_REPO", "/data/jingyu/Flora-TempSeg")),
    )
    parser.add_argument("--profile", choices=("human",), default="human")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    flora_repo = args.flora_repo.expanduser().resolve(strict=True)
    sys.path.insert(0, str(flora_repo))
    _python310_file_digest_compat()

    from batch_utils.api import DataEnginePipeline
    from batch_utils.bootstrap import production as production_adapter
    from batch_utils.config import MODELS, PROFILE_POOLS
    from batch_utils import models as flora_models
    from batch_utils import pipeline as flora_pipeline

    engine = DataEnginePipeline()
    videos, canonical_profile = engine._inputs(
        args.video,
        args.profile,
        None,
        None,
        False,
        args.instruction,
    )
    if args.warmup_runs < 0 or args.measured_runs < 1:
        raise ValueError("warmup-runs must be >= 0 and measured-runs must be >= 1")

    progress_events: list[dict] = []
    runtime_events: list[dict] = []
    started = time.monotonic()

    def progress(fraction: float, desc: str = "") -> None:
        event = {
            "elapsed_s": round(time.monotonic() - started, 3),
            "fraction": fraction,
            "description": desc,
        }
        progress_events.append(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)

    original_runtime_event = flora_models.runtime_event

    def capture_runtime_event(event: str, **values) -> None:
        runtime_events.append(
            {
                "elapsed_s": round(time.monotonic() - started, 3),
                "event": event,
                **values,
            }
        )
        original_runtime_event(event, **values)

    flora_models.runtime_event = capture_runtime_event
    session = flora_models.Session()
    try:
        load_started = time.monotonic()
        model, proc = session.load("tempseg")
        model_load_s = time.monotonic() - load_started
        pool = PROFILE_POOLS[canonical_profile]
        main_video = videos["main"]
        source = production_adapter.AnnotationSource(
            source_id="upload_" + main_video.sha256[:24],
            location=str(main_video.path),
            duration_s=main_video.duration_s,
            fps=float(main_video.rate),
        )
        bound = {
            "candidate_id": "flora_final",
            "checkpoint_sha256": MODELS["tempseg"]["sha256"],
            "model_io_contract": production_adapter.fine_window_profile(
                production_adapter.PROFILES[pool]
            ).model_io_contract,
        }
        runs = []
        for run_index in range(args.warmup_runs + args.measured_runs):
            is_warmup = run_index < args.warmup_runs
            before_events = len(runtime_events)
            run_started = time.monotonic()
            predictor = flora_pipeline.Predictor(
                videos,
                source,
                pool,
                bound,
                model,
                proc,
                args.instruction,
                progress,
            )
            engine_for_run = production_adapter.production_segmenter(
                predictor, bound, pool
            )
            raw_segments = asyncio.run(engine_for_run.segment_async(source))
            run_elapsed_s = time.monotonic() - run_started
            segments = [
                {
                    "start_s": row.span.start_s,
                    "end_s": row.span.end_s,
                    "metadata": dict(row.metadata),
                }
                for row in raw_segments
            ]
            generation_s = sum(
                event.get("elapsed_s", 0.0)
                for event in runtime_events[before_events:]
                if event["event"] == "generation_end"
            )
            runs.append(
                {
                    "run_index": run_index,
                    "warmup": is_warmup,
                    "elapsed_s": round(run_elapsed_s, 3),
                    "generation_s": round(generation_s, 3),
                    "windows": len(predictor.calls),
                    "segments": segments,
                    "window_audit": predictor.calls,
                }
            )
    finally:
        session.close()
        flora_models.runtime_event = original_runtime_event

    total_elapsed_s = time.monotonic() - started
    measured_runs = [row for row in runs if not row["warmup"]]
    elapsed_s = sum(row["elapsed_s"] for row in measured_runs)
    generation_s = sum(row["generation_s"] for row in measured_runs)
    window_count = sum(row["windows"] for row in measured_runs)
    segments = measured_runs[-1]["segments"]
    payload = {
        "schema_version": "flora_tempseg_phase_boundaries_v1",
        "source_video": str(args.video.expanduser().resolve()),
        "source_sha256": videos["main"].sha256,
        "source_duration_s": videos["main"].duration_s,
        "source_fps": str(videos["main"].rate),
        "profile": canonical_profile,
        "models": {
            "tempseg": MODELS["tempseg"],
            "base": MODELS["tempseg_base"],
        },
        "segments": segments,
        "phase_boundaries_s": sorted(
            {round(value, 6) for row in segments for value in (row["start_s"], row["end_s"])}
        ),
        "metrics": {
            "model_load_s": round(model_load_s, 3),
            "measured_elapsed_s": round(elapsed_s, 3),
            "measured_generation_s": round(generation_s, 3),
            "total_elapsed_s": round(total_elapsed_s, 3),
            "windows": window_count,
            "window_hz": round(window_count / elapsed_s, 6) if elapsed_s else None,
            "generation_window_hz": round(window_count / generation_s, 6)
            if generation_s
            else None,
            "video_seconds_per_wall_second": round(videos["main"].duration_s / elapsed_s, 6)
            if elapsed_s
            else None,
        },
        "progress": progress_events,
        "runs": runs,
        "runtime_events": runtime_events,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), **payload["metrics"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
