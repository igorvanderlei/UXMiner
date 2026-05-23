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
MIN_CIRCULARITY = 0.85
MIN_RED_PIXELS = 80

CLICK_DEBOUNCE_SECONDS = 0.10
SPATIAL_DEBOUNCE_PIXELS = 20

DOUBLE_CLICK_WINDOW_SECONDS = 0.25
DOUBLE_CLICK_MAX_DISTANCE_PIXELS = 45

CLICK_BURST_WINDOW_SECONDS = 3.0
CLICK_BURST_MAX_CLICKS = 15



SAVE_DEBUG_FRAMES = True
DEBUG_MAX_FRAMES = 280

CROP_TOP_PIXELS = 180
CROP_BOTTOM_PIXELS = 0
CROP_LEFT_PIXELS = 0
CROP_RIGHT_PIXELS = 0

ENABLE_HOUGH_RED = True
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 35
HOUGH_PARAM1 = 90
HOUGH_PARAM2 = 28
HOUGH_MIN_RADIUS = 22
HOUGH_MAX_RADIUS = 65

MIN_RING_RED_PIXELS = 12
MIN_FILL_RED_RATIO = 0.20

MIN_INNER_STRONG_RED_PIXELS = 120
MIN_INNER_STRONG_RED_RATIO = 0.06
INNER_RADIUS_FACTOR = 0.72

INNER_TEXTURE_RADIUS_FACTOR = 0.65
MAX_INNER_GRAY_STDDEV = 70.0
MIN_INNER_MEAN_SATURATION = 35.0



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

    return frame[top:bottom, left:right], left, top

def strong_red_mask_from_cropped(cropped):
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 80, 80])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 80, 80])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def red_mask_from_frame(frame):
    cropped, offset_x, offset_y = crop_frame(frame)

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 45, 45])
    upper_red_1 = np.array([14, 255, 255])
    lower_red_2 = np.array([166, 45, 45])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return mask, offset_x, offset_y

def evaluate_inner_texture(cropped, x: int, y: int, radius: int) -> dict:
    height, width = cropped.shape[:2]

    inner_radius = max(int(radius * INNER_TEXTURE_RADIUS_FACTOR), 1)

    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.circle(
        mask,
        (int(x), int(y)),
        inner_radius,
        255,
        -1
    )

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    gray_values = gray[mask > 0]
    saturation_values = hsv[:, :, 1][mask > 0]

    if len(gray_values) == 0:
        return {
            "inner_gray_stddev": 999.0,
            "inner_mean_saturation": 0.0
        }

    return {
        "inner_gray_stddev": float(np.std(gray_values)),
        "inner_mean_saturation": float(np.mean(saturation_values))
    }


def contour_candidate(contour, red_pixels: int | None = None) -> dict | None:
    area = cv2.contourArea(contour)

    if area < MIN_CIRCLE_AREA or area > MAX_CIRCLE_AREA:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if circularity < MIN_CIRCULARITY:
        return None

    (x, y), radius = cv2.minEnclosingCircle(contour)

    return {
        "x": float(x),
        "y": float(y),
        "radius": float(radius),
        "area": float(area),
        "circularity": float(circularity),
        "red_pixels": red_pixels or 0,
        "detector": "red_hsv_contour"
    }


def detect_red_circle_contour(frame) -> tuple[bool, dict | None]:
    mask, offset_x, offset_y = red_mask_from_frame(frame)

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
        candidate = contour_candidate(contour, red_pixels=red_pixels)

        if candidate is None:
            continue

        candidate["x"] += offset_x
        candidate["y"] += offset_y

        if best is None or candidate["area"] > best["area"]:
            best = candidate

    if best is None:
        return False, None

    return True, best


def detect_red_circle_hough(frame) -> tuple[bool, dict | None]:
    cropped, offset_x, offset_y = crop_frame(frame)
    mask, _, _ = red_mask_from_frame(frame)
    strong_mask = strong_red_mask_from_cropped(cropped)

    red_pixels = int(cv2.countNonZero(mask))

    if red_pixels < MIN_RED_PIXELS:
        return False, None

    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS
    )

    if circles is None:
        return False, None

    circles = np.round(circles[0, :]).astype("int")

    best = None

    for x, y, radius in circles:
        circle_mask = np.zeros_like(mask)
        cv2.circle(circle_mask, (x, y), radius, 255, 2)

        overlap = cv2.bitwise_and(mask, circle_mask)
        overlap_pixels = int(cv2.countNonZero(overlap))

        # Exige alguma evidência vermelha na borda circular.
        if overlap_pixels < MIN_RING_RED_PIXELS:
            continue

        inner_mask = np.zeros_like(mask)

        inner_radius = max(int(radius * INNER_RADIUS_FACTOR), 1)

        cv2.circle(
            inner_mask,
            (int(x), int(y)),
            inner_radius,
            255,
            -1
        )

        inner_area = int(cv2.countNonZero(inner_mask))

        inner_strong_red = int(
            cv2.countNonZero(
                cv2.bitwise_and(strong_mask, inner_mask)
            )
        )

        inner_strong_red_ratio = (
            inner_strong_red / inner_area
            if inner_area > 0
            else 0
        )

        if (
            inner_strong_red < MIN_INNER_STRONG_RED_PIXELS
            or inner_strong_red_ratio < MIN_INNER_STRONG_RED_RATIO
        ):
            continue

        texture = evaluate_inner_texture(
            cropped=cropped,
            x=int(x),
            y=int(y),
            radius=int(radius)
        )

        if texture["inner_gray_stddev"] > MAX_INNER_GRAY_STDDEV:
            continue

        if texture["inner_mean_saturation"] < MIN_INNER_MEAN_SATURATION:
            continue


        filled_mask = np.zeros_like(mask)
        cv2.circle(filled_mask, (x, y), max(radius - 4, 1), 255, -1)

        filled_area = int(cv2.countNonZero(filled_mask))
        filled_red = int(cv2.countNonZero(cv2.bitwise_and(mask, filled_mask)))

        fill_red_ratio = filled_red / filled_area if filled_area else 0

        if fill_red_ratio < MIN_FILL_RED_RATIO:
            continue

        area_estimate = np.pi * radius * radius

        candidate = {
            "x": float(x + offset_x),
            "y": float(y + offset_y),
            "radius": float(radius),
            "area": float(area_estimate),
            "circularity": 1.0,
            "red_pixels": red_pixels,
            "hough_overlap_pixels": overlap_pixels,
            "detector": "red_hough",
            "inner_strong_red": inner_strong_red,
            "inner_strong_red_ratio": inner_strong_red_ratio,
            "inner_gray_stddev": texture["inner_gray_stddev"],
            "inner_mean_saturation": texture["inner_mean_saturation"]
        }

        if best is None or candidate["hough_overlap_pixels"] > best["hough_overlap_pixels"]:
            best = candidate

    if best is None:
        return False, None

    return True, best

def detect_click_candidate(frame) -> tuple[bool, dict | None]:
#    detected, info = detect_red_circle_contour(frame)

#    if detected:
#        return True, info

    if ENABLE_HOUGH_RED:
        detected, info = detect_red_circle_hough(frame)

        if detected:
            return True, info

    return False, None


def should_register_click(
    clicks: list[dict],
    timestamp: float,
    x: float,
    y: float
) -> bool:
    if not clicks:
        return True

    for last in reversed(clicks[-10:]):
        time_delta = timestamp - last["timestamp_seconds"]

        if time_delta > CLICK_DEBOUNCE_SECONDS:
            break

        dx = x - last["x"]
        dy = y - last["y"]
        spatial_distance = (dx * dx + dy * dy) ** 0.5

        if spatial_distance < SPATIAL_DEBOUNCE_PIXELS:
            return False

    return True


def classify_double_clicks(clicks: list[dict]) -> None:
    double_click_group = 0

    for i in range(1, len(clicks)):
        current = clicks[i]
        previous = clicks[i - 1]

        if current["task_id"] != previous["task_id"]:
            continue

        delta_time = (
            current["timestamp_seconds"]
            - previous["timestamp_seconds"]
        )

        if delta_time < CLICK_DEBOUNCE_SECONDS:
            continue

        if delta_time > DOUBLE_CLICK_WINDOW_SECONDS:
            continue

        dx = current["x"] - previous["x"]
        dy = current["y"] - previous["y"]

        distance = (dx * dx + dy * dy) ** 0.5

        if distance > DOUBLE_CLICK_MAX_DISTANCE_PIXELS:
            continue

        if "double_click_group" not in previous:
            double_click_group += 1

            previous["double_click_group"] = (
                f"DC{double_click_group:04d}"
            )

            previous["click_sequence_in_group"] = 1

        current["double_click_group"] = (
            previous["double_click_group"]
        )

        current["click_sequence_in_group"] = (
            previous["click_sequence_in_group"] + 1
        )

def detect_click_burst(
    task_clicks: list[dict],
    timestamp: float
) -> bool:

    recent = [
        click
        for click in task_clicks
        if (
            timestamp - click["timestamp_seconds"]
            <= CLICK_BURST_WINDOW_SECONDS
        )
    ]

    return len(recent) >= CLICK_BURST_MAX_CLICKS


def process_video(video_path: Path, tasks: list[dict], output_dir: Path) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    failed_tasks = set()
    task_clicks_map = {}

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

        if task["task_id"] in failed_tasks:
            frame_index += 1
            continue

        detected, info = detect_click_candidate(frame)

        task_clicks_map.setdefault(task["task_id"], [])

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
                    "red_pixels": info["red_pixels"],
                    "detector": info["detector"]
                }

                if "hough_overlap_pixels" in info:
                    click["hough_overlap_pixels"] = info["hough_overlap_pixels"]

                task_clicks_map[task["task_id"]].append(click)

                if detect_click_burst(
                    task_clicks_map[task["task_id"]],
                    timestamp
                ):
                    failed_tasks.add(task["task_id"])

                    print(
                        f"[BURST FAILURE] "
                        f"{task['task_id']} "
                        f"abortada por rajada de cliques."
                    )

                    continue

                clicks.append(click)

                if SAVE_DEBUG_FRAMES and debug_count < DEBUG_MAX_FRAMES:
                    annotated = frame.copy()

                    if info["detector"] == "red_hsv_contour":
                        color = (0, 255, 255)
                    elif info["detector"] == "red_hough":
                        color = (0, 165, 255)
                    else:
                        color = (255, 255, 0)

                    cv2.circle(
                        annotated,
                        (int(x), int(y)),
                        int(info["radius"]),
                        color,
                        3
                    )

                    cv2.putText(
                        annotated,
                        f"{click['click_id']} {task['task_id']} {timestamp:.2f}s {info['detector']}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2
                    )

                    debug_path = (
                        debug_dir /
                        f"{click['click_id']}_{task['task_id']}_{timestamp:.2f}_{info['detector']}.png"
                    )
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
            "clicks_red_hsv_contour": sum(1 for c in task_clicks if c["detector"] == "red_hsv_contour"),
            "clicks_red_hough": sum(1 for c in task_clicks if c["detector"] == "red_hough"),
            "clicks_motion_fallback": sum(1 for c in task_clicks if c["detector"] == "motion_fallback"),
            "clicks_per_minute": (
                len(task_clicks) / (task["duration_seconds"] / 60)
                if task["duration_seconds"] > 0
                else 0
            ),
            "detection_failed": (
                len(task_clicks) >= CLICK_BURST_MAX_CLICKS
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
                "spatial_debounce_pixels": SPATIAL_DEBOUNCE_PIXELS,
                "crop": {
                    "top_pixels": CROP_TOP_PIXELS,
                    "bottom_pixels": CROP_BOTTOM_PIXELS,
                    "left_pixels": CROP_LEFT_PIXELS,
                    "right_pixels": CROP_RIGHT_PIXELS
                },
                "red_detector": {
                    "min_circle_area": MIN_CIRCLE_AREA,
                    "max_circle_area": MAX_CIRCLE_AREA,
                    "min_circularity": MIN_CIRCULARITY,
                    "min_red_pixels": MIN_RED_PIXELS
                },
                "hough_detector": {
                    "enabled": ENABLE_HOUGH_RED,
                    "dp": HOUGH_DP,
                    "min_dist": HOUGH_MIN_DIST,
                    "param1": HOUGH_PARAM1,
                    "param2": HOUGH_PARAM2,
                    "min_radius": HOUGH_MIN_RADIUS,
                    "max_radius": HOUGH_MAX_RADIUS
                },
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
        fieldnames = [
            "click_id",
            "task_id",
            "timestamp_seconds",
            "relative_timestamp_seconds",
            "x",
            "y",
            "radius",
            "area",
            "circularity",
            "red_pixels",
            "detector",
            "hough_overlap_pixels",
            "double_click_group",
            "click_sequence_in_group"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for click in clicks:
            row = {key: click.get(key, "") for key in fieldnames}
            writer.writerow(row)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "task_duration_seconds",
                "clicks_count",
                "clicks_red_hsv_contour",
                "clicks_red_hough",
                "clicks_motion_fallback",
                "clicks_per_minute",
                "detection_failed"
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
    classify_double_clicks(clicks)
    summary = summarize_clicks(clicks, tasks)

    save_outputs(video_name, clicks, summary, tasks)

    print("\nResumo por tarefa:")
    for item in summary:
        print(
            f"{item['task_id']} | "
            f"{item['clicks_count']} cliques | "
            f"HSV={item['clicks_red_hsv_contour']} | "
            f"Hough={item['clicks_red_hough']} | "
            f"{item['clicks_per_minute']:.2f} cliques/min"
        )


if __name__ == "__main__":
    main()
