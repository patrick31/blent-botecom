#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run.sh --mac --email EMAIL --question QUESTION
  ./run.sh --mac
  ./run.sh --mac --email EMAIL
  ./run.sh --cuda --email EMAIL --question QUESTION
  ./run.sh --cuda
  ./run.sh --cuda --email EMAIL
  ./run.sh --mac --streamlit
  ./run.sh --mac --test
  ./run.sh --cuda --test
  ./run.sh --cuda --install
  ./run.sh --mac --check
  ./run.sh --cuda --check

Options:
  --mac        Configure le backend MLX pour Mac Apple Silicon.
  --cuda       Configure le backend Transformers pour GPU Nvidia/CUDA.
  --check      Verifie les dependances sans charger le modele, puis quitte.
  --streamlit  Lance l'interface Streamlit au lieu de la CLI.
  --test Lance les tests pytest avec le backend choisi.
  --install    Installe les dependances du backend choisi.
  --email      Email de l'utilisateur authentifie. Si absent, affiche un menu de selection.
  --question Question a poser a l'assistant. Si absent, demarre le mode interactif.
  --help       Affiche cette aide.
EOF
}

backend=""
mode="cli"
install=false
check=false
email=""
question=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mac)
      backend="mac"
      shift
      ;;
    --cuda)
      backend="cuda"
      shift
      ;;
    --streamlit)
      mode="streamlit"
      shift
      ;;
    --test)
      mode="test"
      shift
      ;;
    --install)
      install=true
      shift
      ;;
    --check)
      check=true
      shift
      ;;
    --email)
      email="${2:-}"
      shift 2
      ;;
    --question)
      question="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Option inconnue: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$backend" ]]; then
  echo "Choisissez un backend avec --mac ou --cuda." >&2
  usage
  exit 1
fi

if [[ "$backend" == "mac" ]]; then
  export LLM_BACKEND="mlx"
  export HF_LLM_MODEL_ID="${HF_LLM_MODEL_ID:-mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit}"
  export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-180}"
  export ANALYSIS_MAX_NEW_TOKENS="${ANALYSIS_MAX_NEW_TOKENS:-160}"
  export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
  install_extra=".[dev,ui,mac]"
else
  export LLM_BACKEND="transformers"
  export HF_LLM_MODEL_ID="${HF_LLM_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
  export HF_LOAD_IN_4BIT="${HF_LOAD_IN_4BIT:-true}"
  export HF_DEVICE_MAP="${HF_DEVICE_MAP:-auto}"
  export HF_TORCH_DTYPE="${HF_TORCH_DTYPE:-auto}"
  export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-180}"
  export ANALYSIS_MAX_NEW_TOKENS="${ANALYSIS_MAX_NEW_TOKENS:-160}"
  export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
  install_extra=".[dev,ui,cuda]"
fi

preflight_mac() {
  PYTHONPATH=src python3 - <<'PY'
import importlib.metadata
import sys

from customer_service_bot.llm import is_mlx_available

print("Backend: mlx", flush=True)
try:
    print(f"mlx: {importlib.metadata.version('mlx')}", flush=True)
    print(f"mlx-lm: {importlib.metadata.version('mlx-lm')}", flush=True)
except Exception as exc:
    print(f"Erreur dependances MLX: {exc}", file=sys.stderr)
    sys.exit(1)

if not is_mlx_available():
    print("Erreur: dependances MLX indisponibles.", file=sys.stderr)
    sys.exit(1)
print("Dependances MLX: disponibles", flush=True)
PY
}

preflight_cuda() {
  python3 - <<'PY'
import sys

print("Backend: transformers/cuda", flush=True)
try:
    import torch
    import transformers
    import accelerate
    import bitsandbytes
    print(f"torch: {torch.__version__}", flush=True)
    print(f"transformers: {transformers.__version__}", flush=True)
    print(f"accelerate: {accelerate.__version__}", flush=True)
    print(f"bitsandbytes: {bitsandbytes.__version__}", flush=True)
except Exception as exc:
    print(f"Erreur dependances CUDA: {exc}", file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print("Erreur: CUDA indisponible pour torch.", file=sys.stderr)
    sys.exit(1)

print(f"CUDA devices: {torch.cuda.device_count()}", flush=True)
for index in range(torch.cuda.device_count()):
    print(f"- {index}: {torch.cuda.get_device_name(index)}", flush=True)
PY
}

preflight() {
  if [[ "$backend" == "mac" ]]; then
    preflight_mac
  else
    preflight_cuda
  fi
}

if [[ "$install" == true ]]; then
  python3 -m pip install -e "$install_extra"
  if [[ "$check" == false && "$mode" == "cli" && -z "$email" && -z "$question" ]]; then
    exit 0
  fi
fi

preflight

if [[ "$check" == true ]]; then
  exit 0
fi

if [[ "$mode" == "test" ]]; then
  PYTHONPATH=src python3 -m pytest tests/
  exit 0
fi

if [[ "$mode" == "streamlit" ]]; then
  PYTHONPATH=src streamlit run app_streamlit.py
  exit 0
fi

cli_command=(python3 -m customer_service_bot.cli)
if [[ -n "$email" ]]; then
  cli_command+=(--email "$email")
fi
if [[ -n "$question" ]]; then
  cli_command+=(--question "$question")
fi

PYTHONPATH=src "${cli_command[@]}"
