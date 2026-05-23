from pathlib import Path
import json
import csv
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
CLICKS_DIR = PROJECT_ROOT / "data" / "clicks"

SUPPORTED_EXTENSIONS = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"]

PROCESSING_FPS = 30
MIN_CIRCLE_AREA = 370
MAX_CIRCLE_AREA = 10000
MIN_CIRCULARITY = 0.75
MIN_RED_PIXELS = 80

CLICK_DEBOUNCE_SECONDS = 0.25

SAVE_DEBUG_FRAMES = True
DEBUG_MAX_FRAMES = 80


CROP_TOP_PIXELS = 180
CROP_BOTTOM_PIXELS = 0
CROP_LEFT_PIXELS = 0
CROP_RIGHT_PIXELS = 0


def crop_frame(frame):
    height, width = frame.shape[:2]

    top = CROP_TOP_PIXELS
    bottom = height - CROP_BOTTOM_PIXELS
    left = CROP_LEFT_PIXELS
    right = width - CROP_RIGHT_PIXELS

    if top >= bottom or left >= right:
        raise ValueError(
            f"Crop inválido: frame={width}x{height}, "
            f"top={top}, bottom={bottom}, left={left}, right={right}"
        )

    cropped = frame[top:bottom, left:right]

    return cropped, left, top


def find_video_file() -> Path:
    for pattern in SUPPORTED_EXTENSIONS:
        matches = sorted(RAW_DIR.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Nenhum vídeo encontrado em {RAW_DIR}")


def find_transcript_file(video_name: str) -> Path:
    expected = TRANSCRIPTS_DIR / f"{video_name}_transcript.json"

    if expected.exists():
        return expected

    matches = sorted(TRANSCRIPTS_DIR.glob("*_transcript.json"))

    if not matches:
        raise FileNotFoundError(f"Nenhuma transcrição encontrada em {TRANSCRIPTS_DIR}")

    return matches[0]


def load_tasks(transcript_path: Path) -> list[dict]:
    with transcript_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])

    if not tasks:
        raise RuntimeError("Nenhuma tarefa detectada no arquivo de transcrição.")

    return tasks


def task_for_timestamp(tasks: list[dict], timestamp: float) -> dict | None:
    for task in tasks:
        if task["start"] <= timestamp <= task["end"]:
            return task

    return None

def detect_red_circle(frame) -> tuple[bool, dict | None]:
    cropped_frame, offset_x, offset_y = crop_frame(frame)

    hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

    # Vermelho em HSV fica nas duas pontas do espectro.
    lower_red_1 = np.array([0, 60, 60])
    upper_red_1 = np.array([12, 255, 255])

    lower_red_2 = np.array([168, 60, 60])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    red_pixels = int(cv2.countNonZero(mask))

    if red_pixels < MIN_RED_PIXELS:
        return False, None

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_CIRCLE_AREA or area > MAX_CIRCLE_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < MIN_CIRCULARITY:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)

        x = x + offset_x
        y = y + offset_y

        candidate = {
            "x": float(x),
            "y": float(y),
            "radius": float(radius),
            "area": float(area),
            "circularity": float(circularity),
            "red_pixels": red_pixels
        }

        if best is None or candidate["area"] > best["area"]:
            best = candidate

    if best is None:
        return False, None

    return True, best


def should_register_click(
    clicks: list[dict],
    timestamp: float,
    x: float,
    y: float
) -> bool:

    if not clicks:
        return True

    last = clicks[-1]

    time_delta = timestamp - last["timestamp_seconds"]

    dx = x - last["x"]
    dy = y - last["y"]

    spatial_distance = (dx * dx + dy * dy) ** 0.5

    # Mesmo local + pouco tempo = provavelmente
    # persistência da animação do clique.
    if (
        time_delta < CLICK_DEBOUNCE_SECONDS
        and spatial_distance < 20
    ):
        return False

    return True


def process_video(video_path: Path, tasks: list[dict], output_dir: Path) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if not original_fps or original_fps <= 0:
        original_fps = 30

    frame_interval = max(int(round(original_fps / PROCESSING_FPS)), 1)

    clicks = []
    debug_count = 0
    debug_dir = output_dir / "debug_frames"

    if SAVE_DEBUG_FRAMES:
        debug_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        task = task_for_timestamp(tasks, timestamp)

        if task is None:
            frame_index += 1
            continue

        detected, info = detect_red_circle(frame)

        if detected and info:
            x = info["x"]
            y = info["y"]

            if should_register_click(clicks, timestamp, x, y):
                click = {
                    "click_id": f"C{len(clicks) + 1:04d}",
                    "task_id": task["task_id"],
                    "timestamp_seconds": timestamp,
                    "relative_timestamp_seconds": timestamp - task["start"],
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "radius": round(info["radius"], 2),
                    "area": round(info["area"], 2),
                    "circularity": round(info["circularity"], 4),
                    "red_pixels": info["red_pixels"]
                }

                clicks.append(click)

                if SAVE_DEBUG_FRAMES and debug_count < DEBUG_MAX_FRAMES:
                    annotated = frame.copy()

                    cv2.circle(
                        annotated,
                        (int(x), int(y)),
                        int(info["radius"]),
                        (0, 255, 255),
                        3
                    )

                    cv2.putText(
                        annotated,
                        f"{click['click_id']} {task['task_id']} {timestamp:.2f}s",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )

                    debug_path = debug_dir / f"{click['click_id']}_{task['task_id']}_{timestamp:.2f}.png"
                    cv2.imwrite(str(debug_path), annotated)
                    debug_count += 1

        frame_index += 1

    cap.release()

    return clicks


def summarize_clicks(clicks: list[dict], tasks: list[dict]) -> list[dict]:
    summary = []

    for task in tasks:
        task_clicks = [
            click for click in clicks
            if click["task_id"] == task["task_id"]
        ]

        summary.append({
            "task_id": task["task_id"],
            "task_duration_seconds": task["duration_seconds"],
            "clicks_count": len(task_clicks),
            "clicks_per_minute": (
                len(task_clicks) / (task["duration_seconds"] / 60)
                if task["duration_seconds"] > 0
                else 0
            )
        })

    return summary


def save_outputs(video_name: str, clicks: list[dict], summary: list[dict], tasks: list[dict]) -> None:
    CLICKS_DIR.mkdir(parents=True, exist_ok=True)

    output_json = CLICKS_DIR / f"{video_name}_clicks.json"
    output_csv = CLICKS_DIR / f"{video_name}_clicks.csv"
    summary_csv = CLICKS_DIR / f"{video_name}_click_summary.csv"

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "processing_fps": PROCESSING_FPS,
                "click_debounce_seconds": CLICK_DEBOUNCE_SECONDS,
                "tasks_count": len(tasks),
                "clicks_count": len(clicks),
                "clicks": clicks,
                "summary_by_task": summary
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "click_id",
                "task_id",
                "timestamp_seconds",
                "relative_timestamp_seconds",
                "x",
                "y",
                "radius",
                "area",
                "circularity",
                "red_pixels"
            ]
        )

        writer.writeheader()
        writer.writerows(clicks)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "task_duration_seconds",
                "clicks_count",
                "clicks_per_minute"
            ]
        )

        writer.writeheader()
        writer.writerows(summary)

    print(f"JSON: {output_json}")
    print(f"CSV cliques: {output_csv}")
    print(f"CSV resumo: {summary_csv}")


def main() -> None:
    video_path = find_video_file()
    video_name = video_path.stem.replace(" ", "_")

    transcript_path = find_transcript_file(video_name)
    tasks = load_tasks(transcript_path)

    output_dir = CLICKS_DIR / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Vídeo: {video_path}")
    print(f"Transcrição: {transcript_path}")
    print(f"Tarefas: {len(tasks)}")
    print(f"FPS de processamento: {PROCESSING_FPS}")

    clicks = process_video(video_path, tasks, output_dir)
    summary = summarize_clicks(clicks, tasks)

    save_outputs(video_name, clicks, summary, tasks)

    print("\nResumo por tarefa:")
    for item in summary:
        print(
            f"{item['task_id']} | "
            f"{item['clicks_count']} cliques | "
            f"{item['clicks_per_minute']:.2f} cliques/min"
        )


if __name__ == "__main__":
    main()
