from pathlib import Path
import subprocess
import shutil
import sys
import time
import csv
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
HOLD_DIR = RAW_DIR / "_uxminer_hold"
CROPS_DIR = PROJECT_ROOT / "data" / "crops"

SUPPORTED_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

PIPELINE_STEPS = [
    ("Extract audio and frames", "src/preprocess/extract_basic.py"),
    ("Transcribe and segment tasks", "src/transcription/transcribe.py"),
    ("Extract task frames", "src/vision/extract_keyframes.py"),
    ("Run OCR", "src/ocr/run_ocr_keyframes.py"),
    ("Build states", "src/states/build_states.py"),
    ("Enrich states", "src/states/enrich_states.py"),
    ("Generate protocol", "src/protocol/generate_protocol.py"),
    ("Detect clicks", "src/interactions/detect_clicks.py"),
    ("Generate HTML report", "src/reports/generate_states_html.py"),
]

GENERATED_DIRS_TO_CLEAN = [
    "data/audio",
    "data/audio_chunks",
    "data/frames",
    "data/task_frames",
    "data/outputs",
    "data/ocr",
    "data/states",
    "data/graphs",
    "data/protocols",
    "data/clicks",
    "data/reports",
]


def find_videos() -> list[Path]:
    videos = [
        p for p in RAW_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(videos)


def classify_videos(videos: list[Path]) -> tuple[Path, list[Path]]:
    experts = [v for v in videos if v.stem == "expert"]
    users = [v for v in videos if v.stem.startswith("user_")]

    if not experts:
        raise RuntimeError("Nenhum vídeo expert encontrado em data/raw.")

    if len(experts) > 1:
        raise RuntimeError("Mais de um vídeo expert encontrado.")

    return experts[0], sorted(users)


def clean_generated_dirs() -> None:
    for rel in GENERATED_DIRS_TO_CLEAN:
        path = PROJECT_ROOT / rel

        if path.exists():
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)


def isolate_video(video: Path) -> list[tuple[Path, Path]]:
    HOLD_DIR.mkdir(parents=True, exist_ok=True)

    moved = []

    for other in find_videos():
        if other.resolve() == video.resolve():
            continue

        dst = HOLD_DIR / other.name
        shutil.move(str(other), str(dst))
        moved.append((dst, other))

    return moved


def restore_videos(moved: list[tuple[Path, Path]]) -> None:
    for src, dst in moved:
        if src.exists():
            shutil.move(str(src), str(dst))

    if HOLD_DIR.exists() and not any(HOLD_DIR.iterdir()):
        HOLD_DIR.rmdir()


def run_step(video_stem: str, label: str, script: str) -> dict:
    print(f"\n=== {label} ===")

    start = time.perf_counter()

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script)],
        cwd=PROJECT_ROOT,
        check=True
    )

    end = time.perf_counter()
    elapsed = end - start

    print(f"[TIME] {label}: {elapsed:.2f}s")

    return {
        "video": video_stem,
        "step": label,
        "script": script,
        "elapsed_seconds": round(elapsed, 3)
    }


def run_single_video(video: Path) -> list[dict]:
    print("\n\n########################################")
    print(f"Processing video: {video.name}")
    print("########################################")

    clean_generated_dirs()

    moved = isolate_video(video)

    timings = []
    total_start = time.perf_counter()

    try:
        for label, script in PIPELINE_STEPS:
            timing = run_step(video.stem, label, script)
            timings.append(timing)

    finally:
        restore_videos(moved)

    total_end = time.perf_counter()

    timings.append({
        "video": video.stem,
        "step": "TOTAL",
        "script": "",
        "elapsed_seconds": round(total_end - total_start, 3)
    })

    print("\nProcessing times:")
    for item in timings:
        print(f"- {item['step']}: {item['elapsed_seconds']:.2f}s")

    return timings


def copy_run_outputs(video_stem: str, timings: list[dict]) -> None:
    archive_dir = PROJECT_ROOT / "data" / "runs" / video_stem

    if archive_dir.exists():
        shutil.rmtree(archive_dir)

    archive_dir.mkdir(parents=True, exist_ok=True)

    crop_src = CROPS_DIR / f"{video_stem}.json"
    crop_dst = archive_dir / "app_crop.json"

    if crop_src.exists():
        shutil.copy2(crop_src, crop_dst)
    else:
        with crop_dst.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "app_crop": {
                        "top": 0,
                        "bottom": 0,
                        "left": 0,
                        "right": 0
                    }
                },
                f,
                indent=2,
                ensure_ascii=False
            )

    for rel in GENERATED_DIRS_TO_CLEAN:
        src = PROJECT_ROOT / rel

        if src.exists():
            dst = archive_dir / Path(rel).name
            shutil.copytree(src, dst, dirs_exist_ok=True)

    json_path = archive_dir / "processing_times.json"
    csv_path = archive_dir / "processing_times.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "step",
                "script",
                "elapsed_seconds"
            ]
        )
        writer.writeheader()
        writer.writerows(timings)

    print(f"Run archived at: {archive_dir}")


def run_metrics_for_user(user_stem: str) -> None:
    print(f"\n=== Compare {user_stem} with expert ===")

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src/metrics/compare_with_benchmark.py"),
            "--expert-run", str(PROJECT_ROOT / "data/runs/expert"),
            "--user-run", str(PROJECT_ROOT / "data/runs" / user_stem),
            "--user-id", user_stem,
        ],
        cwd=PROJECT_ROOT,
        check=True
    )


def run_aggregate_metrics() -> None:
    print("\n=== Aggregate user metrics ===")

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src/metrics/aggregate_user_metrics.py"),
        ],
        cwd=PROJECT_ROOT,
        check=True
    )


def save_all_timings(all_timings: list[dict]) -> None:
    runs_dir = PROJECT_ROOT / "data" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = runs_dir / "processing_times_all.csv"
    json_path = runs_dir / "processing_times_all.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "step",
                "script",
                "elapsed_seconds"
            ]
        )
        writer.writeheader()
        writer.writerows(all_timings)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_timings, f, indent=2, ensure_ascii=False)

    print(f"\nAll processing times CSV: {csv_path}")
    print(f"All processing times JSON: {json_path}")


def main() -> None:
    videos = find_videos()
    expert, users = classify_videos(videos)

    print(f"Expert video: {expert.name}")
    print(f"User videos: {[u.name for u in users]}")

    all_timings = []

    expert_timings = run_single_video(expert)
    all_timings.extend(expert_timings)
    copy_run_outputs("expert", expert_timings)

    for user_video in users:
        user_timings = run_single_video(user_video)
        all_timings.extend(user_timings)

        copy_run_outputs(user_video.stem, user_timings)
        run_metrics_for_user(user_video.stem)

    run_aggregate_metrics()
    save_all_timings(all_timings)

    print("\nBatch pipeline completed.")


if __name__ == "__main__":
    main()
