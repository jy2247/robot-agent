"""Build the reusable success-versus-failure critic demo package."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "demo" / "critic-comparison"
DEFAULT_OUTPUT = Path("/data/jliu3292/robot-agent-results/critic-demo-success-vs-failure")
FAILURE_SOURCE = Path("/data/jliu3292/robot-agent-results/vega-pipette-fail-5-three-camera/case-01-composite-global-safety-phase-aligned-freeze10s")
FAILURE_VIDEOS = [
    Path("/data/jliu3292/robot-agent-data/vega-pipette-fail-5/case-01-freeze-10s/vega_fail_01_S2_dropped_ws_bridge_v31_30k_b0828_ego_freeze10s.mp4"),
    Path("/data/jliu3292/robot-agent-data/vega-pipette-fail-5/case-01-freeze-10s/vega_fail_01_S2_dropped_ws_bridge_v31_30k_b0828_left_wrist_freeze10s.mp4"),
    Path("/data/jliu3292/robot-agent-data/vega-pipette-fail-5/case-01-freeze-10s/vega_fail_01_S2_dropped_ws_bridge_v31_30k_b0828_right_wrist_freeze10s.mp4"),
]
SUCCESS_SOURCE_REL = Path("cases/success/inference-freeze10s-rotated")
SUCCESS_VIDEO_REL = [
    Path("cases/success/videos/success_step2_overhead_freeze10s.mp4"),
    Path("cases/success/videos/success_step2_wrist_freeze10s.mp4"),
]


def load_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def fit_frame(frame, width, height):
    canvas = np.full((height, width, 3), (10, 15, 17), dtype=np.uint8)
    if frame is None:
        return canvas
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))))
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def draw_text(image, text, position, scale=.55, color=(225, 235, 234), thickness=1):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def latest_row(rows, time_s):
    result = None
    for row in rows:
        if row["source_time_s"] <= time_s + .01:
            result = row
        else:
            break
    return result


def render_annotated(case, output_path):
    sources = [Path(path) for path in case["absolute_videos"]]
    captures = [cv2.VideoCapture(str(path)) for path in sources]
    source_fps = captures[0].get(cv2.CAP_PROP_FPS) or 30
    frame_counts = [capture.get(cv2.CAP_PROP_FRAME_COUNT) for capture in captures]
    durations = [count / (capture.get(cv2.CAP_PROP_FPS) or source_fps) for capture, count in zip(captures, frame_counts)]
    duration_s = min(durations)
    output_fps = 30.0
    frame_total = int(duration_s * output_fps)
    rows = case["prediction_rows"]
    labels = case["camera_labels"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="critic-demo-") as temporary:
        intermediate = Path(temporary) / "intermediate.mp4"
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (1920, 1080))
        for output_index in range(frame_total):
            time_s = output_index / output_fps
            canvas = np.full((1080, 1920, 3), (7, 11, 13), dtype=np.uint8)
            frames = []
            for camera_index, (capture, path) in enumerate(zip(captures, sources)):
                capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
                ok, frame = capture.read()
                if ok and camera_index in case.get("rotate_180_cameras", []):
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                frames.append(frame if ok else None)

            draw_text(canvas, "ROBOT SAFETY CRITIC", (40, 46), .62, (192, 242, 121), 2)
            draw_text(canvas, "PHASE + GLOBAL OBJECT-STATE MONITOR", (40, 72), .36, (126, 144, 149))
            draw_text(canvas, case["title"].upper(), (1380, 52), .48, (224, 233, 233), 1)
            draw_text(canvas, f"{time_s:05.2f} s", (1785, 52), .48, (126, 144, 149), 1)

            if len(frames) == 2:
                placements = [(40, 105, 820, 820), (880, 105, 430, 820)]
            else:
                placements = [(40, 105, 820, 820), (880, 105, 430, 400), (880, 525, 430, 400)]
            for frame, label, (x, y, width, height) in zip(frames, labels, placements):
                tile = fit_frame(frame, width, height)
                canvas[y:y + height, x:x + width] = tile
                cv2.rectangle(canvas, (x, y), (x + width, y + height), (42, 54, 58), 1)
                cv2.rectangle(canvas, (x + 12, y + 12), (x + 150, y + 42), (9, 14, 16), -1)
                draw_text(canvas, label.upper(), (x + 24, y + 33), .35, (219, 230, 231))

            panel_x = 1350
            cv2.rectangle(canvas, (panel_x, 105), (1880, 925), (14, 20, 23), -1)
            cv2.rectangle(canvas, (panel_x, 105), (1880, 925), (42, 54, 58), 1)
            row = latest_row(rows, time_s)
            if row is None:
                verdict, accent = "WARMING UP", (237, 216, 99)
                phase = pipette = tube = "--"
                confirmation = 0
                action = "CONTINUE"
                confidence = 0
            else:
                failed = row["safety"]["failure_accepted"]
                verdict = "FAILURE DETECTED" if failed else "STATE CONSISTENT"
                accent = (95, 102, 255) if failed else (192, 242, 121)
                phase = row["phase"]["status"].replace("_", " ").upper()
                pipette = row["safety"]["pipette_support"].replace("_", " ").upper()
                tube = row["safety"]["tube_support"].replace("_", " ").upper()
                confirmation = row["safety"]["failure_confirmation_count"]
                action = "STOP" if failed else "CONTINUE"
                confidence = int(row["phase"].get("confidence", 0) * 100)
            cv2.circle(canvas, (1390, 150), 6, accent, -1)
            draw_text(canvas, verdict, (1410, 157), .48, accent, 2)
            draw_text(canvas, "EXPECTED PHASE", (1390, 215), .36, (119, 137, 142))
            draw_text(canvas, "S2  PICK TARGET TUBE", (1390, 250), .58, (229, 237, 236), 1)
            cv2.line(canvas, (1390, 280), (1840, 280), (42, 54, 58), 1)
            draw_text(canvas, "PHASE VERDICT", (1390, 325), .36, (119, 137, 142))
            draw_text(canvas, phase, (1390, 365), .67, (105, 216, 237), 2)
            draw_text(canvas, f"CONFIDENCE  {confidence}%", (1390, 395), .36, (119, 137, 142))
            cv2.line(canvas, (1390, 425), (1840, 425), (42, 54, 58), 1)
            draw_text(canvas, "GLOBAL OBJECT STATE", (1390, 470), .36, (119, 137, 142))
            draw_text(canvas, "PIPETTE", (1390, 525), .38, (119, 137, 142))
            draw_text(canvas, pipette, (1515, 525), .58, accent if pipette in {"WORK SURFACE", "FLOOR"} else (192, 242, 121), 2)
            draw_text(canvas, "TUBE", (1390, 580), .38, (119, 137, 142))
            draw_text(canvas, tube, (1515, 580), .58, accent if tube in {"WORK SURFACE", "FLOOR"} else (105, 216, 237), 2)
            cv2.line(canvas, (1390, 620), (1840, 620), (42, 54, 58), 1)
            draw_text(canvas, "FAILURE CONFIRMATION", (1390, 665), .36, (119, 137, 142))
            for index in range(2):
                color = (95, 102, 255) if index < confirmation else (42, 54, 58)
                cv2.rectangle(canvas, (1390 + index * 230, 695), (1600 + index * 230, 702), color, -1)
            draw_text(canvas, f"{confirmation} / 2", (1768, 665), .4, (220, 230, 230))
            cv2.rectangle(canvas, (1390, 770), (1840, 850), accent, 1)
            draw_text(canvas, "ROBOT ACTION", (1415, 800), .35, (119, 137, 142))
            draw_text(canvas, action, (1415, 834), .72, accent, 2)

            timeline_y = 990
            cv2.line(canvas, (40, timeline_y), (1880, timeline_y), (42, 54, 58), 2)
            for assessment in rows:
                x = int(40 + (assessment["source_time_s"] / duration_s) * 1840)
                color = (95, 102, 255) if assessment["safety"]["failure_accepted"] else (192, 242, 121)
                cv2.circle(canvas, (x, timeline_y), 4, color, -1)
            cursor_x = int(40 + (time_s / duration_s) * 1840)
            cv2.circle(canvas, (cursor_x, timeline_y), 8, (240, 245, 245), -1)
            draw_text(canvas, "ASSESSMENT TIMELINE", (40, 1040), .36, (119, 137, 142))
            writer.write(canvas)
        writer.release()
        for capture in captures:
            capture.release()
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(intermediate),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-video-render", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    success_source = output / SUCCESS_SOURCE_REL
    success_videos = [output / path for path in SUCCESS_VIDEO_REL]
    required = [success_source / "predictions.jsonl", success_source / "summary.json", *success_videos,
                FAILURE_SOURCE / "predictions.jsonl", FAILURE_SOURCE / "summary.json", *FAILURE_VIDEOS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("missing required inputs: " + ", ".join(missing))

    for name in ("index.html", "styles.css", "app.js", "README.md"):
        copy_file(TEMPLATE_DIR / name, output / name)

    failure_dir = output / "cases" / "failure"
    for name in ("predictions.jsonl", "summary.json"):
        copy_file(FAILURE_SOURCE / name, failure_dir / "inference" / name)
    failure_destinations = []
    for source, name in zip(FAILURE_VIDEOS, ("failure_step2_ego_freeze10s.mp4", "failure_step2_left_wrist_freeze10s.mp4", "failure_step2_right_wrist_freeze10s.mp4")):
        destination = failure_dir / "videos" / name
        copy_file(source, destination)
        failure_destinations.append(destination)

    success_rows = load_jsonl(success_source / "predictions.jsonl")
    failure_rows = load_jsonl(failure_dir / "inference" / "predictions.jsonl")
    success_summary = json.loads((success_source / "summary.json").read_text())
    failure_summary = json.loads((failure_dir / "inference" / "summary.json").read_text())
    cases = {
        "success": {
            "title": "Successful tube pick",
            "ground_truth": "success",
            "camera_labels": ["Overhead", "Left wrist"],
            "rotate_180_cameras": [1],
            "absolute_videos": [str(path) for path in success_videos],
            "videos": [{"label": label, "src": str(path.relative_to(output)), "rotate": 180 if index == 1 else 0} for index, (label, path) in enumerate(zip(("OVERHEAD · RGB", "LEFT WRIST"), success_videos))],
            "prediction_rows": success_rows,
            "summary_data": success_summary,
            "safety_mean_ms": success_summary["safety_inference"]["mean_ms"],
            "predictions": str((success_source / "predictions.jsonl").relative_to(output)),
            "summary": str((success_source / "summary.json").relative_to(output)),
            "annotated_video": "exports/success_critic_demo.mp4",
            "story": {"title": "Safe state continuity", "body": "The tube transitions from rack support to hand support while the pipette remains controlled. The global critic observes no forbidden support state across all 18 assessments."},
            "events": [
                {"time": "01.75 S", "title": "Pipette controlled", "detail": "Pipette support: hand. Tube begins rack-supported."},
                {"time": "04.75 S", "title": "Tube acquired", "detail": "Tube support changes to hand without an unsafe intermediate state."},
                {"time": "18.75 S", "title": "No stop issued", "detail": "Safety remains clear through the extended observation period."},
            ],
        },
        "failure": {
            "title": "Dropped pipette",
            "ground_truth": "failure",
            "camera_labels": ["Ego", "Left wrist", "Right wrist"],
            "rotate_180_cameras": [],
            "absolute_videos": [str(path) for path in failure_destinations],
            "videos": [{"label": label, "src": str(path.relative_to(output))} for label, path in zip(("EGO · RGB", "LEFT WRIST", "RIGHT WRIST"), failure_destinations)],
            "prediction_rows": failure_rows,
            "summary_data": failure_summary,
            "safety_mean_ms": failure_summary["safety_inference"]["mean_ms"],
            "predictions": "cases/failure/inference/predictions.jsonl",
            "summary": "cases/failure/inference/summary.json",
            "annotated_video": "exports/failure_critic_demo.mp4",
            "story": {"title": "Cross-phase inconsistency caught", "body": "The phase critic remains in progress, but the global critic sees the pipette persist on the work surface. Two consecutive observations convert visual evidence into a stop decision."},
            "events": [
                {"time": "15.10 S", "title": "Drop becomes visible", "detail": "The pipette leaves controlled hand support."},
                {"time": "17.75 S", "title": "Unsafe state observed", "detail": "Pipette support classified as work surface; confirmation 1 of 2.", "kind": "fail"},
                {"time": "18.75 S", "title": "Failure accepted", "detail": "Persistent unsafe support triggers the robot stop action.", "kind": "fail"},
            ],
        },
    }

    output.joinpath("exports").mkdir(parents=True, exist_ok=True)
    if not args.skip_video_render:
        for key, case in cases.items():
            render_annotated(case, output / case["annotated_video"])
    public_cases = {key: {k: v for k, v in case.items() if k != "absolute_videos"} for key, case in cases.items()}
    payload = {"schema_version": "1.0", "title": "Robot Safety Critic — Success vs Failure", "cases": public_cases}
    (output / "data.js").write_text("window.DEMO_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    report = f"""# Robot Safety Critic demo report

## Direct result

| Case | Ground truth | Assessments | Global safety result | Combined action |
| --- | --- | ---: | --- | --- |
| Successful tube pick | Success | {len(success_rows)} | {success_summary['safety_statuses'].get('safe', 0)}/{len(success_rows)} safe; no accepted failure | Continue |
| Dropped pipette | Failure | {len(failure_rows)} | Failure accepted at {failure_summary['first_safety_failure_source_time_s']:.2f} s | Stop |

Both runs had zero phase-backend and zero safety-backend errors.

## What the comparison demonstrates

The phase critic remained `in_progress` in both examples. The companion global critic separated them by tracking object support across the active phase:

- successful task: tube support transitions from rack to hand without persistent forbidden support;
- failure task: pipette support persists on the work surface for two consecutive assessments, triggering `STOP`.

The failure is visibly present at approximately 15.10 s, first classified as unsafe at 17.75 s, and accepted after confirmation at 18.75 s.

## Interpretation note

This is a behavioral demonstration, not a controlled accuracy benchmark. The successful clip is a real G1 two-camera teleoperation episode; the failure clip is a three-camera VEGA episode. Both are evaluated with the same phase-plus-global-safety logic and model configuration.

## Rebuild

The editable template is in `demo/critic-comparison/` in the robot-agent repository. Run `scripts/build_critic_demo.py` to regenerate this package.
"""
    (output / "REPORT.md").write_text(report)
    print(json.dumps({"output": str(output), "cases": list(cases), "exports": [case["annotated_video"] for case in cases.values()]}, indent=2))


if __name__ == "__main__":
    main()
