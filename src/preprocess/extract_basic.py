from pathlib import Path
import subprocess
import cv2
import json
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

SUPPORTED_EXTENSIONS = [
    "*.mp4",
    "*.mkv",
    "*.mov",
    "*.avi",
    "*.webm"
]


def run_command(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def find_video_file() -> Path:
    for pattern in SUPPORTED_EXTENSIONS:
        matches = sorted(RAW_DIR.glob(pattern))

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Nenhum vídeo encontrado em {RAW_DIR}. "
        f"Extensões suportadas: mp4, mkv, mov, avi, webm."
    )


def safe_stem(video_path: Path) -> str:
    return video_path.stem.replace(" ", "_")


def extract_audio(video_path: Path, audio_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(audio_path)
    ]

    run_command(command)


def extract_frames(video_path: Path, output_dir: Path, fps: int = 2) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(output_dir / "frame_%06d.png")
    ]

    run_command(command)


def get_video_metadata(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    duration = frame_count / fps if fps else 0

    cap.release()

    return {
        "video": str(video_path),
        "filename": video_path.name,
        "extension": video_path.suffix.lower(),
        "fps_original": fps,
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "duration_seconds": duration
    }


def main() -> None:
    video_path = find_video_file()
    video_name = safe_stem(video_path)

    audio_path = AUDIO_DIR / f"{video_name}.wav"
    frames_output_dir = FRAMES_DIR / video_name
    metadata_path = OUTPUT_DIR / f"{video_name}_metadata.json"

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Vídeo encontrado: {video_path}")

    print("Extraindo metadados...")
    metadata = get_video_metadata(video_path)

    print("Extraindo áudio...")
    extract_audio(video_path, audio_path)

    print("Extraindo frames...")
    extract_frames(video_path, frames_output_dir, fps=2)

    metadata["audio"] = str(audio_path)
    metadata["frames_dir"] = str(frames_output_dir)
    metadata["extracted_fps"] = 2

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Concluído.")
    print(f"Áudio: {audio_path}")
    print(f"Frames: {frames_output_dir}")
    print(f"Metadados: {metadata_path}")


if __name__ == "__main__":
    main()
