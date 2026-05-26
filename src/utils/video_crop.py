from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_video_crop(video_stem: str) -> dict:
    path = PROJECT_ROOT / "data" / "crops" / f"{video_stem}.json"

    default = {
        "top": 0,
        "bottom": 0,
        "left": 0,
        "right": 0
    }

    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    crop = default.copy()
    crop.update(data.get("app_crop", data))

    return crop
    
def apply_crop_to_pil(image, crop: dict):
    width, height = image.size

    left = crop.get("left", 0)
    top = crop.get("top", 0)
    right = width - crop.get("right", 0)
    bottom = height - crop.get("bottom", 0)

    return image.crop((left, top, right, bottom))
    
def apply_crop_to_cv(frame, crop: dict):
    height, width = frame.shape[:2]

    top = crop.get("top", 0)
    bottom = height - crop.get("bottom", 0)
    left = crop.get("left", 0)
    right = width - crop.get("right", 0)

    return frame[top:bottom, left:right], left, top
