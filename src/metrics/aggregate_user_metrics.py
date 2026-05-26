from pathlib import Path
import csv
import json
import math
from statistics import mean, median, stdev


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNS_DIR = PROJECT_ROOT / "data" / "runs"
AGGREGATE_DIR = PROJECT_ROOT / "data" / "runs" / "_aggregate_metrics"


NUMERIC_FIELDS = [
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
    "expert_click_events",
    "user_click_events",
    "click_events_delta",
    "user_single_click_events",
    "user_double_click_groups",
]


def parse_value(value):
    if value in (None, "", "N/A"):
        return None

    try:
        return float(value)
    except ValueError:
        return value


def load_user_metric_files() -> list[Path]:
    files = []

    for run_dir in sorted(RUNS_DIR.glob("user_*")):
        metrics_dir = run_dir / "metrics"

        if not metrics_dir.exists():
            continue

        files.extend(sorted(metrics_dir.glob("*_task_metrics.csv")))

    return files


def load_rows(files: list[Path]) -> list[dict]:
    rows = []

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                parsed = {
                    key: parse_value(value)
                    for key, value in row.items()
                }

                rows.append(parsed)

    return rows


def safe_mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]

    if not values:
        return None

    return mean(values)


def safe_median(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]

    if not values:
        return None

    return median(values)


def safe_stdev(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]

    if len(values) < 2:
        return None

    return stdev(values)


def fmt(value):
    if value is None:
        return "N/A"

    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return round(value, 4)

    return value


def group_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {}

    for row in rows:
        task_id = row.get("task_id")
        if not task_id:
            continue

        grouped.setdefault(task_id, []).append(row)

    return grouped


def aggregate_by_task(rows: list[dict]) -> list[dict]:
    grouped = group_by_task(rows)
    output = []

    for task_id, task_rows in sorted(grouped.items()):
        ok_rows = [
            row for row in task_rows
            if row.get("status") == "ok"
        ]

        failed_click_rows = [
            row for row in ok_rows
            if str(row.get("click_detection_failed")).lower() == "true"
        ]

        entry = {
            "task_id": task_id,
            "users_count": len(task_rows),
            "ok_users_count": len(ok_rows),
            "click_detection_failed_count": len(failed_click_rows),
        }

        for field in NUMERIC_FIELDS:
            values = [
                row.get(field)
                for row in ok_rows
                if isinstance(row.get(field), (int, float))
            ]

            entry[f"{field}_mean"] = safe_mean(values)
            entry[f"{field}_median"] = safe_median(values)
            entry[f"{field}_stdev"] = safe_stdev(values)

        output.append(entry)

    return output


def aggregate_overall(rows: list[dict]) -> dict:
    ok_rows = [
        row for row in rows
        if row.get("status") == "ok"
    ]

    users = {
        row.get("user_id")
        for row in rows
        if row.get("user_id")
    }

    tasks = {
        row.get("task_id")
        for row in rows
        if row.get("task_id")
    }

    failed_click_rows = [
        row for row in ok_rows
        if str(row.get("click_detection_failed")).lower() == "true"
    ]

    output = {
        "users_count": len(users),
        "tasks_count": len(tasks),
        "rows_count": len(rows),
        "ok_rows_count": len(ok_rows),
        "click_detection_failed_rows_count": len(failed_click_rows),
    }

    for field in NUMERIC_FIELDS:
        values = [
            row.get(field)
            for row in ok_rows
            if isinstance(row.get(field), (int, float))
        ]

        output[f"{field}_mean"] = safe_mean(values)
        output[f"{field}_median"] = safe_median(values)
        output[f"{field}_stdev"] = safe_stdev(values)

    return output


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: fmt(row.get(key))
                for key in fieldnames
            })


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    files = load_user_metric_files()

    if not files:
        raise RuntimeError("Nenhum arquivo *_task_metrics.csv encontrado.")

    rows = load_rows(files)

    by_task = aggregate_by_task(rows)
    overall = aggregate_overall(rows)

    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)

    save_csv(
        AGGREGATE_DIR / "aggregate_metrics_by_task.csv",
        by_task
    )

    save_json(
        AGGREGATE_DIR / "aggregate_metrics_by_task.json",
        by_task
    )

    save_json(
        AGGREGATE_DIR / "aggregate_metrics_overall.json",
        overall
    )

    save_csv(
        AGGREGATE_DIR / "all_user_task_metrics.csv",
        rows
    )

    print("Aggregate metrics generated.")
    print(AGGREGATE_DIR / "aggregate_metrics_by_task.csv")
    print(AGGREGATE_DIR / "aggregate_metrics_by_task.json")
    print(AGGREGATE_DIR / "aggregate_metrics_overall.json")
    print(AGGREGATE_DIR / "all_user_task_metrics.csv")


if __name__ == "__main__":
    main()
