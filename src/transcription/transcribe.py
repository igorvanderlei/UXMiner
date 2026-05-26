from pathlib import Path
import json
import re
import subprocess
import shutil

from faster_whisper import WhisperModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
AUDIO_CHUNKS_DIR = PROJECT_ROOT / "data" / "audio_chunks"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"

MODEL_SIZE = "small"

SILENCE_NOISE_DB = "-35dB"

SILENCE_MIN_DURATION = 0.70
MIN_CHUNK_DURATION = 8.0
MAX_CHUNK_DURATION = 60.0

ENABLE_DENOISING = False
CHUNK_PADDING_SECONDS = 0.25
DENOISE_FILTER = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25"

MIN_SECONDS_BETWEEN_STARTS = 20


def denoise_audio(audio_path: Path) -> Path:
    if not ENABLE_DENOISING:
        return audio_path

    denoised_dir = AUDIO_CHUNKS_DIR / "_denoised"
    denoised_dir.mkdir(parents=True, exist_ok=True)

    output_path = denoised_dir / f"{audio_path.stem}_denoised.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            "-af", DENOISE_FILTER,
            "-ac", "1",
            "-ar", "16000",
            str(output_path)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    return output_path


def find_audio_file() -> Path:
    matches = sorted(AUDIO_DIR.glob("*.wav"))
    if not matches:
        raise FileNotFoundError(f"Nenhum áudio encontrado em {AUDIO_DIR}")
    return matches[0]


NUMBER_WORDS = {
    "um": "1",
    "uma": "1",
    "dois": "2",
    "duas": "2",
    "três": "3",
    "tres": "3",
    "quatro": "4",
    "cinco": "5",
    "seis": "6",
    "sete": "7",
    "oito": "8",
    "nove": "9",
    "dez": "10",
    "onze": "11",
    "doze": "12",
    "treze": "13",
    "catorze": "14",
    "quatorze": "14",
    "quinze": "15",
    "dezesseis": "16",
    "dezessete": "17",
    "dezoito": "18",
    "dezenove": "19",
    "vinte": "20",
}



def normalize_text(text: str) -> str:
    text = text.lower().strip()

    text = re.sub(r"[.,;:!?]", "", text)

    text = re.sub(r"\benfim\s*,?\s*tarefa\b", "fim tarefa", text)
    text = re.sub(r"\bafim\s*,?\s*tarefa\b", "fim tarefa", text)
    text = re.sub(r"\be\s*,?\s*enfim\s*,?\s*tarefa\b", "fim tarefa", text)
    text = re.sub(r"\bfim\s*,?\s*tarefa\b", "fim tarefa", text)
    text = re.sub(r"\benfim\s*,?\s*tarefa\b", "fim tarefa", text)
    text = re.sub(r"\be\s*,?\s*enfim\s*,?\s*tarefa\b", "fim tarefa", text)
    text = re.sub(r"\bin[ií]cio\s*,?\s*tarefa\b", "início tarefa", text)
    text = re.sub(r"\bin[ií]cio\s+da\s+reforma\b", "início tarefa 1", text)
    text = re.sub(r"\bparefa\b", "tarefa", text)
    text = re.sub(r"\btarifa\b", "tarefa", text)

    for word, number in NUMBER_WORDS.items():
        text = re.sub(
            rf"\b{re.escape(word)}\b",
            number,
            text
        )

    text = re.sub(r"\s+", " ", text)
    return text


def is_trivial_hallucination(text: str) -> bool:
    normalized = normalize_text(text)
    return normalized in {"e", "é", "a", "o", "ã", "hum", "hmm"} or len(normalized) <= 2


def format_task_id(number: int) -> str:
    return f"T{number:02d}"


def get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return float(result.stdout.strip())


def detect_silences(audio_path: Path) -> list[tuple[float, float]]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", str(audio_path),
            "-af", f"silencedetect=noise={SILENCE_NOISE_DB}:d={SILENCE_MIN_DURATION}",
            "-f", "null",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stderr = result.stderr

    starts = []
    silences = []

    for line in stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)

        if start_match:
            starts.append(float(start_match.group(1)))

        if end_match and starts:
            start = starts.pop(0)
            end = float(end_match.group(1))
            silences.append((start, end))

    return silences


def build_chunk_ranges(duration: float, silences: list[tuple[float, float]]) -> list[tuple[float, float]]:
    split_points = []

    for start, end in silences:
        midpoint = (start + end) / 2.0
        if 0 < midpoint < duration:
            split_points.append(midpoint)

    split_points = sorted(set(split_points))

    ranges = []
    current_start = 0.0

    for point in split_points:
        if point - current_start >= MIN_CHUNK_DURATION:
            ranges.append((current_start, point))
            current_start = point

    if duration - current_start >= MIN_CHUNK_DURATION:
        ranges.append((current_start, duration))

    # Garante limite máximo aproximado, mesmo se não houver silêncio suficiente.
    final_ranges = []

    for start, end in ranges:
        if end - start <= MAX_CHUNK_DURATION:
            final_ranges.append((start, end))
            continue

        cursor = start
        while cursor < end:
            next_end = min(cursor + MAX_CHUNK_DURATION, end)
            if next_end - cursor >= MIN_CHUNK_DURATION:
                final_ranges.append((cursor, next_end))
            cursor = next_end

    return final_ranges


def extract_chunk(audio_path: Path, output_path: Path, start: float, end: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", "16000",
            str(output_path)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


def prepare_audio_chunks(audio_path: Path) -> list[dict]:
    processed_audio_path = denoise_audio(audio_path)

    duration = get_audio_duration(processed_audio_path)
    silences = detect_silences(processed_audio_path)
    ranges = build_chunk_ranges(duration, silences)

    chunk_dir = AUDIO_CHUNKS_DIR / audio_path.stem

    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)

    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunks = []

    for idx, (start, end) in enumerate(ranges, start=1):
        chunk_path = chunk_dir / f"chunk_{idx:04d}.wav"

        padded_start = max(0.0, start - CHUNK_PADDING_SECONDS)
        padded_end = min(duration, end + CHUNK_PADDING_SECONDS)

        extract_chunk(processed_audio_path, chunk_path, padded_start, padded_end)

        chunks.append({
            "chunk_id": f"C{idx:04d}",
            "path": chunk_path,
            "start": padded_start,
            "end": padded_end,
            "duration": padded_end - padded_start,
            "original_start": start,
            "original_end": end
        })

    return chunks


def estimate_timestamp(seg: dict, char_index: int) -> float:
    text = seg["text"]
    ratio = char_index / max(len(text), 1)
    return seg["start"] + ratio * (seg["end"] - seg["start"])


def extract_numbered_markers_inside_segment(seg: dict) -> list[dict]:
    text_original = seg["text"]
    text = normalize_text(text_original)

    marker_patterns = [
        ("start", re.compile(r"\bin[ií]cio\s*(?:do|da)?\s*,?\s*tarefa\s*(\d+)\b")),
        ("start", re.compile(r"\bcome[cç]o\s*(?:do|da)?\s*,?\s*tarefa\s*(\d+)\b")),
        ("end", re.compile(r"\bfim\s*,?\s*tarefa\s*(\d+)\b")),
        ("end", re.compile(r"\bfim\s*(?:do|da)?\s*tarefa\s*(\d+)\b")),
    ]

    found = []

    for marker_type, pattern in marker_patterns:
        for match in pattern.finditer(text):
            task_id = format_task_id(int(match.group(1)))
            timestamp = estimate_timestamp(seg, match.start())

            found.append({
                "type": marker_type,
                "task_id": task_id,
                "timestamp": timestamp,
                "text": text_original,
                "source": "numbered_marker_inside_segment",
                "match_text": match.group(0),
                "char_start": match.start(),
                "char_end": match.end()
            })

    found.sort(key=lambda item: (item["timestamp"], item["char_start"]))
    return found


def is_end_marker(text: str) -> bool:
    text = normalize_text(text)

    patterns = [
        r"\bfim\s+da\s+tarefa\b",
        r"\bfim\s+de\s+tarefa\b",
        r"\btarefa\s+encerrada\b",
        r"\btarefa\s+finalizada\b",
        r"\bessa\s+foi\s+a\b.*\btarefa\b",
        r"\bo\s+qu[aã]o\s+dif[ií]cil\s+foi\b",
        r"\bqu[aã]o\s+dif[ií]cil\s+foi\b",
        r"\bo\s+tempo\s+acabou\b",
        r"\btempo\s+acabou\b",
        r"\btempo\b.*\btarefa\b.*\bencerrou\b",
        r"\btempo\b.*\btarefa\b.*\bacabou\b",
        r"\btempo\b.*\bencerrou\b",
        r"\btempo\b.*\bacabou\b",
        r"\btempo\s+para\s+fazer\b.*\bencerrou\b",
        r"\bfim\b"
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def is_start_marker(text: str) -> bool:
    text = normalize_text(text)

    negative_patterns = [
        r"\bessa\s+foi\b",
        r"\bo\s+qu[aã]o\s+dif[ií]cil\b",
        r"\bqu[aã]o\s+dif[ií]cil\b",
        r"\btempo\b.*\bacabou\b",
        r"\btempo\b.*\bencerrou\b",
        r"\bcompletar\s+a\s+tarefa\b",
    ]

    if any(re.search(pattern, text) for pattern in negative_patterns):
        return False

    patterns = [
        r"\bpode\s+come[cç]ar\b",
        r"\bpode\s+iniciar\b",
        r"\bin[ií]cio\s+da\s+tarefa\b",
        r"\bin[ií]cio\s+de\s+tarefa\b",
        r"\bcome[cç]o\s+da\s+tarefa\b",
        r"\bcome[cç]o\s+de\s+tarefa\b",
        r"\btarefa\s+iniciada\b",
        r"\bvamos\s+para\s+a\s+tarefa\b",
        r"\bvamos\s+para\s+a\s+\w+\s+tarefa\b",
        r"\ba\s+gente\s+vai\s+para\s+a\s+tarefa\b",
        r"\ba\s+gente\s+vai\s+para\s+a\s+\w+\s+tarefa\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def recently_started(markers: list[dict], current_start: float) -> bool:
    for marker in reversed(markers):
        if marker["type"] == "start":
            return (current_start - marker["timestamp"]) < MIN_SECONDS_BETWEEN_STARTS
    return False


def add_marker(markers: list[dict], marker: dict) -> None:
    for existing in markers:
        if (
            existing["type"] == marker["type"]
            and existing["task_id"] == marker["task_id"]
            and abs(existing["timestamp"] - marker["timestamp"]) < 0.25
        ):
            return

    markers.append(marker)


def detect_task_markers(segments: list[dict], video_duration: float | None = None) -> list[dict]:
    markers = []
    open_task_id = None
    next_task_number = 1

    for seg in segments:
        text = normalize_text(seg["text"])

        numbered_markers = extract_numbered_markers_inside_segment(seg)

#        if numbered_markers:
#            for marker in numbered_markers:
#                add_marker(markers, marker)
#            continue

        if numbered_markers:
            for marker in numbered_markers:
                marker_type = marker["type"]
                task_id = marker["task_id"]

                if marker_type == "start":
                    if open_task_id is not None:
                        add_marker(markers, {
                            "type": "end",
                            "task_id": open_task_id,
                            "timestamp": marker["timestamp"],
                            "text": marker.get("text", ""),
                            "source": "implicit_end_by_next_start"
                        })

                    add_marker(markers, marker)
                    open_task_id = task_id

                    try:
                        task_number = int(task_id.replace("T", ""))
                        next_task_number = max(next_task_number, task_number + 1)
                    except ValueError:
                        pass

                elif marker_type == "end":
                    add_marker(markers, marker)

                    if open_task_id == task_id:
                        open_task_id = None

            continue

        if is_start_marker(text):
            if recently_started(markers, seg["start"]):
                continue

            new_task_id = format_task_id(next_task_number)

            if open_task_id is not None:
                add_marker(markers, {
                    "type": "end",
                    "task_id": open_task_id,
                    "timestamp": seg["start"],
                    "text": seg["text"],
                    "source": "implicit_end_by_next_start"
                })

            add_marker(markers, {
                "type": "start",
                "task_id": new_task_id,
                "timestamp": seg["end"],
                "text": seg["text"],
                "source": "start_sequential"
            })

            open_task_id = new_task_id
            next_task_number += 1
            continue

        if is_end_marker(text):
            if open_task_id is not None:
                add_marker(markers, {
                    "type": "end",
                    "task_id": open_task_id,
                    "timestamp": seg["start"],
                    "text": seg["text"],
                    "source": "end_marker"
                })

                open_task_id = None

            continue

    if open_task_id is not None:
        add_marker(markers, {
            "type": "end",
            "task_id": open_task_id,
            "timestamp": video_duration,
            "text": "",
            "source": "implicit_end_eof"
        })

    markers.sort(key=lambda item: (item["timestamp"], item["type"]))
    return markers


def build_task_segments(markers: list[dict]) -> list[dict]:
    tasks = {}

    for marker in markers:
        task_id = marker["task_id"]

        if task_id not in tasks:
            tasks[task_id] = {"task_id": task_id}

        if marker["type"] == "start":
            if "start" not in tasks[task_id]:
                tasks[task_id]["start"] = marker["timestamp"]
                tasks[task_id]["start_marker_text"] = marker.get("text", "")
                tasks[task_id]["start_marker_source"] = marker.get("source", "")
                tasks[task_id]["start_match_text"] = marker.get("match_text", "")

        elif marker["type"] == "end":
            if "start" in tasks[task_id] and marker["timestamp"] > tasks[task_id]["start"]:
                if "end" not in tasks[task_id]:
                    tasks[task_id]["end"] = marker["timestamp"]
                    tasks[task_id]["end_marker_text"] = marker.get("text", "")
                    tasks[task_id]["end_marker_source"] = marker.get("source", "")
                    tasks[task_id]["end_match_text"] = marker.get("match_text", "")

    output = []

    for task_id, data in sorted(tasks.items()):
        if "start" not in data or "end" not in data:
            continue

        duration = data["end"] - data["start"]

        if duration <= 0:
            continue

        data["duration_seconds"] = duration
        output.append(data)

    return output


def transcribe_chunks(model: WhisperModel, chunks: list[dict]) -> tuple[list[dict], str, float]:
    all_segments = []
    detected_language = "pt"
    total_duration = 0.0

    for chunk in chunks:
        print(
            f"Transcribing {chunk['chunk_id']} "
            f"({chunk['start']:.2f}s–{chunk['end']:.2f}s)..."
        )

        segments_generator, info = model.transcribe(
            str(chunk["path"]),
            language="pt",
            beam_size=1,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 120,
                "speech_pad_ms": 80
            },
            no_speech_threshold=0.55,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False
        )

        detected_language = info.language
        total_duration = max(total_duration, chunk["end"])

        for segment in segments_generator:
            text = segment.text.strip()

            if is_trivial_hallucination(text):
                continue

            all_segments.append({
                "start": chunk["start"] + segment.start,
                "end": chunk["start"] + segment.end,
                "text": text,
                "chunk_id": chunk["chunk_id"]
            })

    all_segments.sort(key=lambda item: item["start"])

    return all_segments, detected_language, total_duration


def main() -> None:
    audio_path = find_audio_file()

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing chunks based on silence...")
    chunks = prepare_audio_chunks(audio_path)
    print(f"Generated chunks: {len(chunks)}")

    print("Loading Whisper model...")
    model = WhisperModel(
        MODEL_SIZE,
        device="cuda",
        compute_type="float16"
    )

    segments, language, duration = transcribe_chunks(model, chunks)

    markers = detect_task_markers(segments, video_duration=duration)
    tasks = build_task_segments(markers)

    transcript_data = {
        "language": language,
        "duration": duration,
        "segments": segments,
        "markers": markers,
        "tasks": tasks,
        "chunking": {
            "enabled": True,
            "denoising_enabled": ENABLE_DENOISING,
            "denoise_filter": DENOISE_FILTER if ENABLE_DENOISING else None,
            "chunk_padding_seconds": CHUNK_PADDING_SECONDS,
            "silence_noise_db": SILENCE_NOISE_DB,
            "silence_min_duration": SILENCE_MIN_DURATION,
            "min_chunk_duration": MIN_CHUNK_DURATION,
            "max_chunk_duration": MAX_CHUNK_DURATION,
            "chunks_count": len(chunks)
        }
    }

    output_path = TRANSCRIPTS_DIR / f"{audio_path.stem}_transcript.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)

    print("Completed.")
    print(f"Transcript saved to: {output_path}")

    print("\nDetected markers:\n")
    for marker in markers:
        print(
            f"{marker['type']} | "
            f"{marker['task_id']} | "
            f"{marker['timestamp']:.2f}s | "
            f"{marker.get('source', '')} | "
            f"{marker.get('match_text', marker.get('text', ''))}"
        )

    print("\nDetected tasks:\n")
    for task in tasks:
        print(
            f"{task['task_id']} | "
            f"{task['duration_seconds']:.2f}s | "
            f"start: {task.get('start_marker_source', '')} | "
            f"end: {task.get('end_marker_source', '')}"
        )

if __name__ == "__main__":
    main()
