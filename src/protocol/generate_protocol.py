from pathlib import Path
import json
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATES_DIR = PROJECT_ROOT / "data" / "states"
PROTOCOLS_DIR = PROJECT_ROOT / "data" / "protocols"

TIMEOUT_ALPHA = 20
TIMEOUT_MIN_SECONDS = 180
TIMEOUT_MAX_SECONDS = 600


def find_states_json() -> Path:
    semantic_matches = sorted(STATES_DIR.glob("*_semantic_states.json"))

    if semantic_matches:
        return semantic_matches[0]

    matches = sorted(STATES_DIR.glob("*_states.json"))

    if not matches:
        raise FileNotFoundError(
            f"Nenhum arquivo de estados encontrado em {STATES_DIR}"
        )

    return matches[0]


def calculate_timeout(expert_duration: float) -> float:
    timeout = expert_duration * TIMEOUT_ALPHA
    timeout = max(timeout, TIMEOUT_MIN_SECONDS)
    timeout = min(timeout, TIMEOUT_MAX_SECONDS)
    return timeout


def build_protocol(states_data: dict) -> dict:
    protocol = {
        "benchmark": {
            "source_states_json": states_data.get("source_ocr_json"),
            "states_fps": states_data.get("states_fps"),
            "state_consolidation": states_data.get("state_consolidation", {}),
            "timeout": {
                "alpha": TIMEOUT_ALPHA,
                "min_seconds": TIMEOUT_MIN_SECONDS,
                "max_seconds": TIMEOUT_MAX_SECONDS
            }
        },
        "tasks": [],
        "states": []
    }

    state_ids_added = set()

    for task in states_data["tasks"]:
        task_states = task["states"]
        task_transitions = task["transitions"]

        expert_sequence = [state["state_id"] for state in task_states]
        final_state = expert_sequence[-1] if expert_sequence else None

        task_entry = {
            "id": task["task_id"],
            "label": task["task_id"],
            "expert_duration_seconds": round(task["duration_seconds"], 3),
            "timeout_seconds": round(calculate_timeout(task["duration_seconds"]), 3),
            "expert_frames_count": task.get("frames_count"),
            "expert_states_count": task["states_count"],
            "expert_transitions_count": task["transitions_count"],
            "expert_sequence": expert_sequence,
            "final_state": final_state,
            "transitions": [
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "duration_seconds": round(edge["duration_seconds"], 3),
                    "start_timestamp": round(edge["start_timestamp"], 3),
                    "end_timestamp": round(edge["end_timestamp"], 3)
                }
                for edge in task_transitions
            ]
        }

        protocol["tasks"].append(task_entry)

        for state in task_states:
            state_id = state["state_id"]

            if state_id in state_ids_added:
                continue

            semantic_label = (
                state.get("semantic_label")
                or state.get("ocr_summary")
                or state_id
            )

            semantic_summary = (
                state.get("semantic_summary")
                or state.get("ocr_summary")
                or ""
            )

            semantic_keywords = (
                state.get("semantic_keywords")
                or state.get("ocr_summary", "").split()[:8]
            )

            protocol["states"].append({
                "id": state_id,
                "label": semantic_label,
                "representative_frame": state["representative_frame"],
                "timestamp_seconds": round(state["timestamp_seconds"], 3),
                "relative_timestamp_seconds": round(
                    state["relative_timestamp_seconds"],
                    3
                ),
                "ocr_fingerprint": state["ocr_fingerprint"],
                "ocr_summary": state["ocr_summary"],
                "semantic_label": semantic_label,
                "semantic_summary": semantic_summary,
                "semantic_keywords": semantic_keywords,
                "input_context": state.get("input_context", {}),
                "state_detection": state.get("state_detection", {})
            })

            state_ids_added.add(state_id)

    return protocol


def main() -> None:
    states_json = find_states_json()

    with states_json.open("r", encoding="utf-8") as f:
        states_data = json.load(f)

    PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)

    protocol = build_protocol(states_data)

    video_name = (
        states_json.name
        .replace("_semantic_states.json", "")
        .replace("_states.json", "")
    )

    output_path = PROTOCOLS_DIR / f"{video_name}_protocol.yaml"

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            protocol,
            f,
            allow_unicode=True,
            sort_keys=False
        )

    print("Concluído.")
    print(f"Protocolo gerado: {output_path}")

    print("\nTarefas no protocolo:")
    for task in protocol["tasks"]:
        print(
            f"{task['id']} | "
            f"duração especialista: {task['expert_duration_seconds']}s | "
            f"timeout: {task['timeout_seconds']}s | "
            f"sequência: {' -> '.join(task['expert_sequence'])}"
        )

    print("\nEstados no protocolo:")
    for state in protocol["states"]:
        print(
            f"{state['id']} | "
            f"{state['semantic_label']} | "
            f"{state.get('semantic_summary', '')}"
        )


if __name__ == "__main__":
    main()
