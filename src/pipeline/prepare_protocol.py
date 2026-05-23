from pathlib import Path
import subprocess
import sys
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_PATHS = [
    DATA_DIR / "audio",
    DATA_DIR / "frames",
    DATA_DIR / "outputs",
    DATA_DIR / "transcripts",
    DATA_DIR / "keyframes",
    DATA_DIR / "ocr",
    DATA_DIR / "states",
    DATA_DIR / "graphs",
    DATA_DIR / "protocols"
]

STEPS = [
    ("Extração básica", "src/preprocess/extract_basic.py"),
    ("Transcrição", "src/transcription/transcribe.py"),
    ("Keyframes por tarefa", "src/vision/extract_keyframes.py"),
    ("OCR dos keyframes", "src/ocr/run_ocr_keyframes.py"),
    ("Estados e transições", "src/states/build_states.py"),
    ("Enriquecimento semântico", "src/states/enrich_states.py"),
    ("Geração do protocolo", "src/protocol/generate_protocol.py"),
]


def clean_outputs() -> None:
    print("\nLimpando saídas anteriores...\n")

    for path in OUTPUT_PATHS:
        if path.exists():
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)

    print("Limpeza concluída.")


def run_step(name: str, script: str) -> None:
    script_path = PROJECT_ROOT / script

    print("\n" + "=" * 70)
    print(f"ETAPA: {name}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Falha na etapa: {name}")


def main() -> None:
    print("Preparando protocolo a partir do vídeo gabarito...")

    clean_outputs()

    for name, script in STEPS:
        run_step(name, script)

    print("\n" + "=" * 70)
    print("PROTOCOLO GERADO COM SUCESSO")
    print("=" * 70)

    print("\nArquivo gerado:")
    print(PROJECT_ROOT / "data" / "protocols")


if __name__ == "__main__":
    main()
