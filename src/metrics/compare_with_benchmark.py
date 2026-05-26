from pathlib import Path
import argparse
import csv
import json
import math
import re
from difflib import SequenceMatcher

import cv2


SUCCESS_THRESHOLD = 0.75
INCONCLUSIVE_THRESHOLD = 0.55

VISUAL_WEIGHT = 0.45
TEXT_WEIGHT = 0.55

DEFAULT_APP_CROP = {
    "top": 0,
    "bottom": 0,
    "left": 0,
    "right": 0
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_app_crop(run_dir: Path) -> dict:
    path = run_dir / "app_crop.json"

    if not path.exists():
        return DEFAULT_APP_CROP.copy()

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    crop = DEFAULT_APP_CROP.copy()
    crop.update(data)
    return crop


def find_one(base: Path, pattern: str) -> Path:
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found: {base}/{pattern}")
    return matches[0]


def find_states_file(run_dir: Path) -> Path:
    states_dir = run_dir / "states"
    semantic = sorted(states_dir.glob("*_semantic_states.json"))
    if semantic:
        return semantic[0]
    return find_one(states_dir, "*_states.json")


def find_clicks_file(run_dir: Path) -> Path | None:
    clicks_dir = run_dir / "clicks"
    matches = sorted(clicks_dir.glob("*_clicks.json"))
    return matches[0] if matches else None


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúâêôãõç]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def keyword_similarity(a: list[str], b: list[str]) -> float:
    set_a = {normalize_text(x) for x in a if normalize_text(x)}
    set_b = {normalize_text(x) for x in b if normalize_text(x)}

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def final_state_text_similarity(expert_state: dict, user_state: dict) -> float:
    label_sim = text_similarity(
        expert_state.get("semantic_label", ""),
        user_state.get("semantic_label", "")
    )

    summary_sim = text_similarity(
        expert_state.get("semantic_summary", ""),
        user_state.get("semantic_summary", "")
    )

    ocr_sim = text_similarity(
        expert_state.get("ocr_summary", ""),
        user_state.get("ocr_summary", "")
    )

    keyword_sim = keyword_similarity(
        expert_state.get("semantic_keywords", []),
        user_state.get("semantic_keywords", [])
    )

    return (
        0.30 * label_sim
        + 0.30 * summary_sim
        + 0.25 * ocr_sim
        + 0.15 * keyword_sim
    )


def apply_crop(img, crop: dict):
    h, w = img.shape[:2]

    top = int(crop.get("top", 0))
    bottom_crop = int(crop.get("bottom", 0))
    left = int(crop.get("left", 0))
    right_crop = int(crop.get("right", 0))

    bottom = h - bottom_crop
    right = w - right_crop

    top = max(0, min(top, h))
    bottom = max(0, min(bottom, h))
    left = max(0, min(left, w))
    right = max(0, min(right, w))

    if top >= bottom or left >= right:
        return img

    return img[top:bottom, left:right]


def image_similarity(
    path_a: str,
    path_b: str,
    crop_a: dict,
    crop_b: dict
) -> float | None:
    img_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

    if img_a is None or img_b is None:
        return None

    img_a = apply_crop(img_a, crop_a)
    img_b = apply_crop(img_b, crop_b)

    size = (320, 180)

    img_a = cv2.resize(img_a, size)
    img_b = cv2.resize(img_b, size)

    hist_a = cv2.calcHist([img_a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([img_b], [0], None, [64], [0, 256])

    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    correlation = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


def classify_success(score, visual=None, text=None):
    if visual is not None and visual >= 0.95:
        return "success"

    if score is None:
        return "inconclusive"

    if score >= SUCCESS_THRESHOLD:
        return "success"

    if score >= INCONCLUSIVE_THRESHOLD:
        return "inconclusive"

    return "failure"


def final_state_similarity(
    expert_task: dict,
    user_task: dict,
    expert_crop: dict,
    user_crop: dict
) -> dict:
    expert_states = expert_task.get("states", [])
    user_states = user_task.get("states", [])

    if not expert_states or not user_states:
        return {
            "final_state_similarity_score": None,
            "final_state_visual_similarity": None,
            "final_state_text_similarity": None,
            "task_success_auto": "inconclusive",
            "task_success_reviewed": "",
            "final_state_expert_id": "",
            "final_state_user_id": "",
            "final_state_expert_label": "",
            "final_state_user_label": "",
        }

    expert_final = expert_states[-1]
    user_final = user_states[-1]

    visual = image_similarity(
        expert_final.get("representative_frame", ""),
        user_final.get("representative_frame", ""),
        expert_crop,
        user_crop
    )

    text = final_state_text_similarity(expert_final, user_final)

    score = text if visual is None else VISUAL_WEIGHT * visual + TEXT_WEIGHT * text

    return {
        "final_state_similarity_score": score,
        "final_state_visual_similarity": visual,
        "final_state_text_similarity": text,
        "task_success_auto": classify_success(score, visual, text),
        "task_success_reviewed": "",
        "final_state_expert_id": expert_final.get("state_id", ""),
        "final_state_user_id": user_final.get("state_id", ""),
        "final_state_expert_label": expert_final.get("semantic_label", ""),
        "final_state_user_label": user_final.get("semantic_label", ""),
    }


def task_map(states_data: dict) -> dict:
    return {
        task["task_id"]: task
        for task in states_data.get("tasks", [])
    }


def clicks_by_task(clicks_data: dict | None) -> dict:
    if not clicks_data:
        return {}

    output = {}

    for click in clicks_data.get("clicks", []):
        task_id = click.get("task_id")
        if task_id:
            output.setdefault(task_id, []).append(click)

    return output


def failed_click_tasks(clicks_data: dict | None) -> set[str]:
    if not clicks_data:
        return set()

    return {
        row["task_id"]
        for row in clicks_data.get("summary_by_task", [])
        if row.get("detection_failed")
    }


def count_double_click_groups(clicks: list[dict]) -> int:
    return len({
        click.get("double_click_group")
        for click in clicks
        if click.get("double_click_group")
    })


def count_single_click_events(clicks: list[dict]) -> int:
    return sum(1 for click in clicks if not click.get("double_click_group"))


def sequence_similarity(expert_sequence: list[str], user_sequence: list[str]) -> float:
    if not expert_sequence and not user_sequence:
        return 1.0
    if not expert_sequence or not user_sequence:
        return 0.0

    m = len(expert_sequence)
    n = len(user_sequence)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if expert_sequence[i] == user_sequence[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[m][n] / max(m, n)


def percent_delta(user_value: float, expert_value: float) -> float | None:
    if expert_value == 0:
        return None
    return ((user_value - expert_value) / expert_value) * 100.0


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return round(value, 4)
    return value


def build_metrics(
    expert_states: dict,
    user_states: dict,
    expert_clicks: dict | None,
    user_clicks: dict | None,
    user_id: str,
    expert_crop: dict,
    user_crop: dict
) -> list[dict]:
    expert_tasks = task_map(expert_states)
    user_tasks = task_map(user_states)

    expert_clicks_map = clicks_by_task(expert_clicks)
    user_clicks_map = clicks_by_task(user_clicks)

    expert_failed_clicks = failed_click_tasks(expert_clicks)
    user_failed_clicks = failed_click_tasks(user_clicks)

    rows = []

    all_task_ids = sorted(set(expert_tasks.keys()) | set(user_tasks.keys()))

    for task_id in all_task_ids:
        expert_task = expert_tasks.get(task_id)
        user_task = user_tasks.get(task_id)

        if expert_task is None:
            rows.append({
                "user_id": user_id,
                "task_id": task_id,
                "status": "missing_in_expert"
            })
            continue

        if user_task is None:
            rows.append({
                "user_id": user_id,
                "task_id": task_id,
                "status": "missing_in_user"
            })
            continue

        expert_duration = float(expert_task.get("duration_seconds", 0))
        user_duration = float(user_task.get("duration_seconds", 0))

        expert_sequence = [
            state["state_id"]
            for state in expert_task.get("states", [])
        ]

        user_sequence = [
            state["state_id"]
            for state in user_task.get("states", [])
        ]

        sim = sequence_similarity(expert_sequence, user_sequence)

        expert_task_clicks = expert_clicks_map.get(task_id, [])
        user_task_clicks = user_clicks_map.get(task_id, [])

        click_detection_failed = (
            task_id in expert_failed_clicks
            or task_id in user_failed_clicks
        )

        if click_detection_failed:
            expert_click_events = None
            user_click_events = None
            click_events_delta = None
            user_single_click_events = None
            user_double_click_groups = None
        else:
            expert_click_events = len(expert_task_clicks)
            user_click_events = len(user_task_clicks)
            click_events_delta = user_click_events - expert_click_events
            user_single_click_events = count_single_click_events(user_task_clicks)
            user_double_click_groups = count_double_click_groups(user_task_clicks)

        final_similarity = final_state_similarity(
            expert_task=expert_task,
            user_task=user_task,
            expert_crop=expert_crop,
            user_crop=user_crop
        )

        rows.append({
            "user_id": user_id,
            "task_id": task_id,
            "status": "ok",

            "task_success_auto": final_similarity["task_success_auto"],
            "task_success_reviewed": final_similarity["task_success_reviewed"],
            "final_state_similarity_score": final_similarity["final_state_similarity_score"],
            "final_state_visual_similarity": final_similarity["final_state_visual_similarity"],
            "final_state_text_similarity": final_similarity["final_state_text_similarity"],
            "final_state_expert_id": final_similarity["final_state_expert_id"],
            "final_state_user_id": final_similarity["final_state_user_id"],
            "final_state_expert_label": final_similarity["final_state_expert_label"],
            "final_state_user_label": final_similarity["final_state_user_label"],

            "expert_duration_seconds": expert_duration,
            "user_duration_seconds": user_duration,
            "duration_delta_seconds": user_duration - expert_duration,
            "duration_delta_percent": percent_delta(user_duration, expert_duration),

            "expert_states_count": expert_task.get("states_count"),
            "user_states_count": user_task.get("states_count"),
            "states_delta": user_task.get("states_count", 0) - expert_task.get("states_count", 0),

            "expert_transitions_count": expert_task.get("transitions_count"),
            "user_transitions_count": user_task.get("transitions_count"),
            "transitions_delta": user_task.get("transitions_count", 0) - expert_task.get("transitions_count", 0),

            "sequence_similarity_lcs": sim,

            "click_detection_failed": click_detection_failed,
            "expert_click_events": expert_click_events,
            "user_click_events": user_click_events,
            "click_events_delta": click_events_delta,
            "user_single_click_events": user_single_click_events,
            "user_double_click_groups": user_double_click_groups,
        })

    return rows


def save_metrics(rows: list[dict], output_dir: Path, user_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{user_id}_task_metrics.json"
    csv_path = output_dir / f"{user_id}_task_metrics.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "user_id",
        "task_id",
        "status",

        "task_success_auto",
        "task_success_reviewed",
        "final_state_similarity_score",
        "final_state_visual_similarity",
        "final_state_text_similarity",
        "final_state_expert_id",
        "final_state_user_id",
        "final_state_expert_label",
        "final_state_user_label",

        "expert_duration_seconds",
        "user_duration_seconds",
        "duration_delta_seconds",
        "duration_delta_percent",

        "expert_states_count",
        "user_states_count",
        "states_delta",

        "expert_transitions_count",
        "user_transitions_count",
        "transitions_delta",

        "sequence_similarity_lcs",

        "click_detection_failed",
        "expert_click_events",
        "user_click_events",
        "click_events_delta",
        "user_single_click_events",
        "user_double_click_groups",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})

    print(f"Task metrics JSON: {json_path}")
    print(f"Task metrics CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--expert-run", required=True)
    parser.add_argument("--user-run", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    expert_run = Path(args.expert_run)
    user_run = Path(args.user_run)

    expert_states = load_json(find_states_file(expert_run))
    user_states = load_json(find_states_file(user_run))

    expert_clicks_file = find_clicks_file(expert_run)
    user_clicks_file = find_clicks_file(user_run)

    expert_clicks = load_json(expert_clicks_file) if expert_clicks_file else None
    user_clicks = load_json(user_clicks_file) if user_clicks_file else None

    expert_crop = load_app_crop(expert_run)
    user_crop = load_app_crop(user_run)

    rows = build_metrics(
        expert_states=expert_states,
        user_states=user_states,
        expert_clicks=expert_clicks,
        user_clicks=user_clicks,
        user_id=args.user_id,
        expert_crop=expert_crop,
        user_crop=user_crop
    )

    output_dir = Path(args.output_dir) if args.output_dir else user_run / "metrics"

    save_metrics(rows, output_dir, args.user_id)


if __name__ == "__main__":
    main()
