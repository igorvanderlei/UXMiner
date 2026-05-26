from pathlib import Path
import json
import requests
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATES_DIR = PROJECT_ROOT / "data" / "states"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def find_states_json() -> Path:
    matches = sorted(STATES_DIR.glob("*_states.json"))

    # Evita pegar arquivo já enriquecido.
    matches = [
        p for p in matches
        if not p.name.endswith("_semantic_states.json")
    ]

    if not matches:
        raise FileNotFoundError(
            f"Nenhum arquivo *_states.json encontrado em {STATES_DIR}"
        )

    return matches[0]


def build_prompt(ocr_text: str, input_context: list[str]) -> str:
    input_section = ""

    if input_context:
        input_section = f"""
Texto detectado em campos ou entradas da interface:
{", ".join(input_context)}
""".strip()

    return f"""
Você receberá OCR ruidoso de uma interface gráfica.

Objetivo:
Extrair um identificador semântico curto do estado funcional da interface.

Regras obrigatórias:
- TODOS os campos devem estar em português do Brasil;
- nunca use inglês;
- ignore navegador, abas, menus do sistema e URLs;
- ignore ruídos evidentes do OCR;
- `etiqueta`: máximo 3 palavras;
- `resumo`: máximo 12 palavras;
- `palavras_chave`: 3 a 6 termos;
- prefira termos funcionais da interface;
- considere o contexto textual informado, se existir;
- responda SOMENTE JSON válido.

Formato:
{{
  "etiqueta": "",
  "resumo": "",
  "palavras_chave": []
}}

{input_section}

OCR:
{ocr_text}
""".strip()


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"Resposta não contém JSON válido: {text}")

    return json.loads(match.group(0))


def enrich_with_ollama(ocr_text: str, input_context: list[str]) -> dict:
    prompt = build_prompt(ocr_text, input_context)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    data = response.json()
    raw_response = data.get("response", "")

    parsed = extract_json(raw_response)

    etiqueta = str(parsed.get("etiqueta", "")).strip()
    resumo = str(parsed.get("resumo", "")).strip()
    palavras_chave = parsed.get("palavras_chave", [])

    if not isinstance(palavras_chave, list):
        palavras_chave = []

    palavras_chave = [
        str(item).strip()
        for item in palavras_chave
        if str(item).strip()
    ]

    return {
        "semantic_label": etiqueta,
        "semantic_summary": resumo,
        "semantic_keywords": palavras_chave,
        "semantic_raw_response": raw_response
    }


def main() -> None:
    states_json = find_states_json()

    with states_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"States file: {states_json}")
    print(f"Ollama model: {MODEL_NAME}")

    for task in data["tasks"]:
        print(f"\nEnriching {task['task_id']}...")

        for state in task["states"]:
            state_id = state["state_id"]
            ocr_text = state.get("ocr_text", "")

            input_context = (
                state
                .get("input_context", {})
                .get("detected_text", [])
            )

            print(f"  {state_id}...", end=" ", flush=True)

            try:
                semantic = enrich_with_ollama(
                    ocr_text=ocr_text,
                    input_context=input_context
                )

                state.update(semantic)

                print(semantic["semantic_label"])

            except Exception as e:
                print(f"ERROR: {e}")

                fallback_keywords = (
                    state.get("ocr_summary", "").split()[:6]
                )

                state.update({
                    "semantic_label": state.get("ocr_summary", state_id),
                    "semantic_summary": "",
                    "semantic_keywords": fallback_keywords,
                    "semantic_error": str(e)
                })

    output_path = STATES_DIR / states_json.name.replace(
        "_states.json",
        "_semantic_states.json"
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\nCompleted.")
    print(f"Enriched states: {output_path}")


if __name__ == "__main__":
    main()
