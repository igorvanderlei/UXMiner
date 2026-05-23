from pathlib import Path
import json

from PIL import Image
import pytesseract


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
OCR_DIR = PROJECT_ROOT / "data" / "ocr"

CROP_TOP_PIXELS = 180
CROP_BOTTOM_PIXELS = 0
CROP_LEFT_PIXELS = 0
CROP_RIGHT_PIXELS = 0


def find_task_frames_json() -> Path:
    matches = sorted(OUTPUT_DIR.glob("*_task_frames.json"))

    if not matches:
        raise FileNotFoundError(
            f"Nenhum arquivo *_task_frames.json encontrado em {OUTPUT_DIR}"
        )

    return matches[0]


def crop_image(image: Image.Image) -> Image.Image:
    width, height = image.size

    left = CROP_LEFT_PIXELS
    top = CROP_TOP_PIXELS
    right = width - CROP_RIGHT_PIXELS
    bottom = height - CROP_BOTTOM_PIXELS

    if left >= right or top >= bottom:
        raise ValueError(
            f"Crop inválido: imagem={width}x{height}, "
            f"left={left}, top={top}, right={right}, bottom={bottom}"
        )

    return image.crop((left, top, right, bottom))


def run_ocr(image_path: Path) -> str:
    image = Image.open(image_path)

    cropped = crop_image(image)

    text = pytesseract.image_to_string(
        cropped,
        lang="por+eng",
        config="--psm 6"
    )

    return text.strip()


def main() -> None:
    task_frames_json = find_task_frames_json()

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

        print(f"OCR da {task['task_id']}...")

        for frame in task["frames"]:
            image_path = Path(frame["frame_file"])

            text = run_ocr(image_path)

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
                f"{len(text)} caracteres"
            )

        output_tasks.append(task_output)

    output = {
        "source_task_frames_json": str(task_frames_json),
        "ocr_crop": {
            "top_pixels": CROP_TOP_PIXELS,
            "bottom_pixels": CROP_BOTTOM_PIXELS,
            "left_pixels": CROP_LEFT_PIXELS,
            "right_pixels": CROP_RIGHT_PIXELS
        },
        "tasks_count": len(output_tasks),
        "tasks": output_tasks
    }

    video_name = task_frames_json.name.replace("_task_frames.json", "")
    output_path = OCR_DIR / f"{video_name}_ocr.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Concluído.")
    print(f"Saída OCR: {output_path}")


if __name__ == "__main__":
    main()
