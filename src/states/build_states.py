from pathlib import Path
import json
import re
import hashlib
import shutil

import cv2
from skimage.metrics import structural_similarity as ssim


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OCR_DIR = PROJECT_ROOT / "data" / "ocr"
STATES_DIR = PROJECT_ROOT / "data" / "states"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
KEYFRAMES_DIR = PROJECT_ROOT / "data" / "keyframes"

VISUAL_SSIM_THRESHOLD = 0.92
TEXT_JACCARD_THRESHOLD = 0.75
MIN_ADDED_TOKENS_FOR_INPUT = 1
STABLE_FRAMES = 2

STOPWORDS = {
    "mozilla", "firefox", "arquivo", "editar", "exibir", "histórico",
    "favoritos", "ferramentas", "ajuda", "https", "www", "com", "br",
    "x", "a", "o", "e", "de", "do", "da", "dos", "das", "em", "para"
}


def find_ocr_json() -> Path:
    matches = sorted(OCR_DIR.glob("*_ocr.json"))
    if not matches:
        raise FileNotFoundError(f"Nenhum arquivo OCR encontrado em {OCR_DIR}")
    return matches[0]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s\*]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> set[str]:
    tokens = normalize_text(text).split()
    return {t for t in tokens if len(t) >= 2 and t not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fingerprint(text: str) -> str:
    cleaned = normalize_text(text)
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()[:10]


def summarize_text(text: str, max_words: int = 12) -> str:
    tokens = [t for t in normalize_text(text).split() if t not in STOPWORDS]
    return " ".join(tokens[:max_words])


def load_gray_resized(path: Path, width: int = 640):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Não foi possível ler imagem: {path}")

    height = int(image.shape[0] * (width / image.shape[1]))
    resized = cv2.resize(image, (width, height))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def visual_similarity(path_a: Path, path_b: Path) -> float:
    a = load_gray_resized(path_a)
    b = load_gray_resized(path_b)
    return float(ssim(a, b))


def detect_masked_input(text: str) -> bool:
    return bool(re.search(r"\*{3,}", text))


def added_terms(previous_text: str, current_text: str) -> list[str]:
    previous_tokens = tokenize(previous_text)
    current_tokens = tokenize(current_text)

    added = sorted(current_tokens - previous_tokens)

    return [
        term for term in added
        if len(term) >= 2
    ]


def copy_representative_frame(src: Path, dst_dir: Path, state_id: str) -> Path:
    dst = dst_dir / f"{state_id}_{src.name}"
    shutil.copy2(src, dst)
    return dst


def should_create_candidate(previous_state: dict, frame: dict) -> tuple[bool, dict]:
    previous_frame = Path(previous_state["representative_frame"])
    current_frame = Path(frame["image_file"])

    visual_sim = visual_similarity(previous_frame, current_frame)

    prev_tokens = tokenize(previous_state.get("ocr_text", ""))
    curr_tokens = tokenize(frame.get("ocr_text", ""))

    text_sim = jaccard(prev_tokens, curr_tokens)
    added = added_terms(previous_state.get("ocr_text", ""), frame.get("ocr_text", ""))

    visual_change = visual_sim < VISUAL_SSIM_THRESHOLD
    textual_change = (
        text_sim < TEXT_JACCARD_THRESHOLD
        or len(added) >= MIN_ADDED_TOKENS_FOR_INPUT
    )

    evidence = {
        "visual_similarity": visual_sim,
        "text_similarity": text_sim,
        "added_terms": added,
        "visual_change": visual_change,
        "textual_change": textual_change,
        "masked_input_detected": detect_masked_input(frame.get("ocr_text", ""))
    }

    return visual_change or textual_change, evidence


def consolidate_task_states(
    task: dict,
    output_frame_dir: Path,
    state_counter_start: int
):
    frames = task["frames"]

    states = []
    transitions = []
    state_counter = state_counter_start

    if not frames:
        return states, transitions, state_counter

    def create_state(frame: dict, evidence: dict | None = None) -> dict:
        nonlocal state_counter

        state_id = f"S{state_counter:04d}"
        state_counter += 1

        copied_frame = copy_representative_frame(
            Path(frame["image_file"]),
            output_frame_dir,
            state_id
        )

        ocr_text = frame.get("ocr_text", "")

        state = {
            "state_id": state_id,
            "task_id": task["task_id"],
            "frame_id": frame["frame_id"],
            "timestamp_seconds": frame["timestamp_seconds"],
            "relative_timestamp_seconds": frame["relative_timestamp_seconds"],
            "representative_frame": str(copied_frame),
            "ocr_fingerprint": fingerprint(ocr_text),
            "ocr_summary": summarize_text(ocr_text),
            "ocr_text": ocr_text,
            "state_detection": evidence or {
                "reason": "initial_state"
            },
            "input_context": {
                "detected_text": [],
                "masked_input_detected": detect_masked_input(ocr_text)
            }
        }

        if evidence and evidence.get("added_terms"):
            state["input_context"]["detected_text"] = evidence["added_terms"]

        return state

    first_state = create_state(frames[0])
    states.append(first_state)

    candidate = None
    stable_count = 0

    for frame in frames[1:]:
        previous_state = states[-1]

        is_candidate, evidence = should_create_candidate(previous_state, frame)

        if not is_candidate:
            candidate = None
            stable_count = 0
            continue

        # Mudança visual forte vira estado imediatamente.
        # Ex.: troca de tela, abertura de listagem, modal, resultado etc.
        if evidence.get("visual_change"):
            new_state = create_state(
                frame,
                {
                    **evidence,
                    "reason": "visual_change"
                }
            )

            states.append(new_state)
            candidate = None
            stable_count = 0
            continue

        # Mudança apenas textual exige estabilidade.
        # Ex.: digitação em input, preenchimento de campo etc.
        current_signature = fingerprint(frame.get("ocr_text", ""))

        if candidate is None or candidate["signature"] != current_signature:
            candidate = {
                "frame": frame,
                "evidence": evidence,
                "signature": current_signature
            }
            stable_count = 1
        else:
            stable_count += 1
            candidate["frame"] = frame
            candidate["evidence"] = evidence

        if stable_count >= STABLE_FRAMES:
            new_state = create_state(
                candidate["frame"],
                {
                    **candidate["evidence"],
                    "reason": "stable_textual_change",
                    "stable_frames": stable_count
                }
            )

            states.append(new_state)
            candidate = None
            stable_count = 0

    for source, target in zip(states, states[1:]):
        transitions.append({
            "task_id": task["task_id"],
            "source": source["state_id"],
            "target": target["state_id"],
            "start_timestamp": source["timestamp_seconds"],
            "end_timestamp": target["timestamp_seconds"],
            "duration_seconds": target["timestamp_seconds"] - source["timestamp_seconds"]
        })

    return states, transitions, state_counter


def build_states_and_transitions(data: dict) -> dict:
    all_tasks = []
    global_state_counter = 1

    video_name = Path(data["source_task_frames_json"]).name.replace(
        "_task_frames.json",
        ""
    )

    frame_output_dir = KEYFRAMES_DIR / video_name

    if frame_output_dir.exists():
        shutil.rmtree(frame_output_dir)

    frame_output_dir.mkdir(parents=True, exist_ok=True)

    for task in data["tasks"]:
        states, transitions, global_state_counter = consolidate_task_states(
            task=task,
            output_frame_dir=frame_output_dir,
            state_counter_start=global_state_counter
        )

        all_tasks.append({
            "task_id": task["task_id"],
            "start": task["start"],
            "end": task["end"],
            "duration_seconds": task["duration_seconds"],
            "frames_count": task["frames_count"],
            "states_count": len(states),
            "transitions_count": len(transitions),
            "states": states,
            "transitions": transitions
        })

    return {
        "source_ocr_json": data["source_task_frames_json"],
        "tasks_count": len(all_tasks),
        "states_fps": 2.0,
        "state_consolidation": {
            "visual_ssim_threshold": VISUAL_SSIM_THRESHOLD,
            "text_jaccard_threshold": TEXT_JACCARD_THRESHOLD,
            "stable_frames": STABLE_FRAMES,
            "min_added_tokens_for_input": MIN_ADDED_TOKENS_FOR_INPUT
        },
        "tasks": all_tasks
    }


def main() -> None:
    ocr_json = find_ocr_json()

    with ocr_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    STATES_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)

    result = build_states_and_transitions(data)

    video_name = ocr_json.name.replace("_ocr.json", "")

    states_output = STATES_DIR / f"{video_name}_states.json"
    edges_output = GRAPHS_DIR / f"{video_name}_edges.csv"

    with states_output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    with edges_output.open("w", encoding="utf-8") as f:
        f.write("task_id,source,target,duration_seconds,start_timestamp,end_timestamp\n")

        for task in result["tasks"]:
            for edge in task["transitions"]:
                f.write(
                    f"{edge['task_id']},"
                    f"{edge['source']},"
                    f"{edge['target']},"
                    f"{edge['duration_seconds']},"
                    f"{edge['start_timestamp']},"
                    f"{edge['end_timestamp']}\n"
                )

    print("Concluído.")
    print(f"Estados: {states_output}")
    print(f"Arestas CSV: {edges_output}")

    for task in result["tasks"]:
        print(
            f"{task['task_id']} | "
            f"{task['frames_count']} frames | "
            f"{task['states_count']} estados | "
            f"{task['transitions_count']} transições"
        )

        for state in task["states"]:
            print(
                f"  {state['state_id']} | "
                f"{state['relative_timestamp_seconds']:.2f}s | "
                f"{state['ocr_summary']}"
            )

            input_text = state.get("input_context", {}).get("detected_text", [])
            if input_text:
                print(f"    input_context: {input_text}")

        for transition in task["transitions"]:
            print(
                f"  {transition['source']} -> {transition['target']} | "
                f"{transition['duration_seconds']:.2f}s"
            )


if __name__ == "__main__":
    main()
