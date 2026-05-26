from pathlib import Path
import json

from PIL import Image
import pytesseract

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.video_crop import load_video_crop, apply_crop_to_pil


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
OCR_DIR = PROJECT_ROOT / "data" / "ocr"

def find_task_frames_json() -> Path:
    matches = sorted(OUTPUT_DIR.glob("*_task_frames.json"))

    if not matches:
        raise FileNotFoundError(
            f"Nenhum arquivo *_task_frames.json encontrado em {OUTPUT_DIR}"
        )

    return matches[0]


def run_ocr(image_path: Path, crop: dict) -> str:
    image = Image.open(image_path)

    cropped = apply_crop_to_pil(image, crop)

    text = pytesseract.image_to_string(
        cropped,
        lang="por+eng",
        config="--psm 6"
    )

    return text.strip()


def main() -> None:
    task_frames_json = find_task_frames_json()

    video_stem = task_frames_json.stem.replace("_task_frames", "")
    crop = load_video_crop(video_stem)

    print(f"Video stem: {video_stem}")
    print(f"OCR crop: {crop}")

    with task_frames_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    OCR_DIR.mkdir(parents=True, exist_ok=True)

    output_tasks = []

    for task in data["tasks"]:
        task_output = {
            "task_id": task["task_id"],
            "start": task["start"],
            "end": task["end"],
            "duration_seconds": task["duration_seconds"],
            "frames_count": task["frames_count"],
            "frames": []
        }

        print(f"OCR for {task['task_id']}...")

        for frame in task["frames"]:
            image_path = Path(frame["frame_file"])

            text = run_ocr(image_path, crop)

            task_output["frames"].append({
                "frame_id": f"{task['task_id']}_{frame['source_frame'].replace('.png', '')}",
                "task_id": task["task_id"],
                "source_frame": frame["source_frame"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "relative_timestamp_seconds": frame["relative_timestamp_seconds"],
                "image_file": str(image_path),
                "ocr_text": text
            })

            print(
                f"  {frame['source_frame']} | "
                f"{frame['relative_timestamp_seconds']:.2f}s | "
                f"{len(text)} characters"
            )

        output_tasks.append(task_output)

    output = {
        "source_task_frames_json": str(task_frames_json),
        "ocr_crop": crop,
        "tasks_count": len(output_tasks),
        "tasks": output_tasks
    }

    video_name = task_frames_json.name.replace("_task_frames.json", "")
    output_path = OCR_DIR / f"{video_name}_ocr.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Completed.")
    print(f"OCR output: {output_path}")


if __name__ == "__main__":
    main()
