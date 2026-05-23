from pathlib import Path
import json
import shutil
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
TASK_FRAMES_DIR = PROJECT_ROOT / "data" / "task_frames"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

EXTRACTED_FPS = 2.0


def find_frames_dir() -> Path:
    dirs = sorted([p for p in FRAMES_DIR.iterdir() if p.is_dir()])

    if not dirs:
        raise FileNotFoundError(f"Nenhum diretório de frames encontrado em {FRAMES_DIR}")

    return dirs[0]


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
        raise RuntimeError(
            f"Nenhuma tarefa encontrada em {transcript_path}. "
            f"Rode primeiro a transcrição e confirme os marcadores."
        )

    return tasks


def frame_index_from_name(frame_path: Path) -> int:
    match = re.search(r"frame_(\d+)\.png", frame_path.name)

    if not match:
        raise ValueError(f"Nome de frame inválido: {frame_path.name}")

    return int(match.group(1))


def timestamp_from_frame(frame_path: Path) -> float:
    frame_index = frame_index_from_name(frame_path)
    return (frame_index - 1) / EXTRACTED_FPS


def frame_belongs_to_task(timestamp: float, task: dict) -> bool:
    return task["start"] <= timestamp <= task["end"]


def copy_task_frame(src: Path, dst_dir: Path, task_id: str) -> Path:
    dst = dst_dir / f"{task_id}_{src.name}"
    shutil.copy2(src, dst)
    return dst


def extract_frames_for_task(
    task: dict,
    frame_paths: list[Path],
    output_dir: Path
) -> list[dict]:
    task_id = task["task_id"]

    task_frames = [
        frame_path
        for frame_path in frame_paths
        if frame_belongs_to_task(timestamp_from_frame(frame_path), task)
    ]

    output = []

    for frame_path in task_frames:
        copied_path = copy_task_frame(frame_path, output_dir, task_id)
        timestamp_seconds = timestamp_from_frame(frame_path)

        output.append({
            "task_id": task_id,
            "source_frame": frame_path.name,
            "frame_file": str(copied_path),
            "timestamp_seconds": timestamp_seconds,
            "relative_timestamp_seconds": timestamp_seconds - task["start"]
        })

    return output


def main() -> None:
    frames_dir = find_frames_dir()
    video_name = frames_dir.name

    transcript_path = find_transcript_file(video_name)
    tasks = load_tasks(transcript_path)

    output_dir = TASK_FRAMES_DIR / video_name

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.png"))

    if not frame_paths:
        raise FileNotFoundError(f"Nenhum frame PNG encontrado em {frames_dir}")

    print(f"Diretório de frames: {frames_dir}")
    print(f"Transcrição: {transcript_path}")
    print(f"Tarefas encontradas: {len(tasks)}")

    output_tasks = []
    total_task_frames = 0

    for task in tasks:
        frames = extract_frames_for_task(
            task=task,
            frame_paths=frame_paths,
            output_dir=output_dir
        )

        total_task_frames += len(frames)

        output_tasks.append({
            "task_id": task["task_id"],
            "start": task["start"],
            "end": task["end"],
            "duration_seconds": task["duration_seconds"],
            "frames_count": len(frames),
            "frames": frames
        })

        print(
            f"{task['task_id']} | "
            f"{task['duration_seconds']:.2f}s | "
            f"{len(frames)} frames da tarefa"
        )

    output_json = OUTPUT_DIR / f"{video_name}_task_frames.json"

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "frames_dir": str(frames_dir),
                "transcript": str(transcript_path),
                "task_frames_dir": str(output_dir),
                "extracted_fps": EXTRACTED_FPS,
                "tasks_count": len(tasks),
                "total_task_frames": total_task_frames,
                "tasks": output_tasks
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print("Concluído.")
    print(f"Total de frames em tarefas: {total_task_frames}")
    print(f"Saída: {output_json}")
    print(f"Imagens: {output_dir}")


if __name__ == "__main__":
    main()
