"""Préflight pytest pour les tests qui chargent le vrai LLM.

Les tests du projet ne mockent pas le modèle. On vérifie donc la configuration
avant de lancer pytest pour échouer vite avec un message clair si le backend
Mac/CUDA n'est pas explicitement configuré.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def pytest_cmdline_main(config: pytest.Config) -> int | None:
    """Bloque les tests LLM si le backend demandé n'est pas utilisable."""

    backend = os.getenv("LLM_BACKEND")
    model_id = os.getenv("HF_LLM_MODEL_ID", "")

    if backend not in {"mlx", "transformers"}:
        return _fail_preflight(
            "Configuration LLM manquante pour pytest.\n"
            "Les tests chargent le vrai LLM: exportez le backend avant de lancer pytest.\n\n"
            "Mac Apple Silicon:\n"
            '  export LLM_BACKEND="mlx"\n'
            '  export HF_LLM_MODEL_ID="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"\n'
            "  pytest\n\n"
            "Plateforme cible Nvidia/CUDA:\n"
            '  export LLM_BACKEND="transformers"\n'
            '  export HF_LLM_MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"\n'
            '  export HF_DEVICE_MAP="auto"\n'
            '  export HF_TORCH_DTYPE="auto"\n'
            '  export HF_LOAD_IN_4BIT="true"\n'
            "  pytest"
        )

    if backend == "mlx":
        return _preflight_mlx(model_id)

    return _preflight_cuda(model_id)


def _fail_preflight(message: str) -> int:
    """Affiche une seule erreur lisible, sans le doublon de pytest.exit."""

    print(message, file=sys.stderr)
    return 2


def _preflight_mlx(model_id: str) -> int | None:
    """Vérifie la configuration minimale attendue pour le backend MLX."""

    if not model_id.startswith("mlx-community/"):
        return _fail_preflight(
            "Configuration pytest incohérente: LLM_BACKEND=mlx nécessite un modèle MLX.\n"
            "Utilisez par exemple:\n"
            '  export HF_LLM_MODEL_ID="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"'
        )

    missing = [package for package in ("mlx", "mlx_lm") if importlib.util.find_spec(package) is None]
    if missing:
        return _fail_preflight(
            "Dépendances MLX absentes pour pytest: "
            f"{', '.join(missing)}.\n"
            'Installez le backend Mac avec: pip install -e ".[dev,ui,mac]"'
        )

    return None


def _preflight_cuda(model_id: str) -> int | None:
    """Vérifie la configuration minimale attendue pour le backend CUDA."""

    if model_id.startswith("mlx-community/"):
        return _fail_preflight(
            "Configuration pytest incohérente: LLM_BACKEND=transformers ne peut pas charger un modèle MLX.\n"
            "Utilisez le modèle cible CUDA:\n"
            '  export HF_LLM_MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"'
        )

    try:
        import torch
    except ImportError:
        return _fail_preflight('PyTorch est absent. Installez le backend CUDA avec: pip install -e ".[dev,ui,cuda]"')

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return _fail_preflight(
            "Configuration pytest incohérente: LLM_BACKEND=transformers exige une GPU Nvidia/CUDA disponible.\n"
            "Aucun fallback CPU n'est utilisé pour ce projet. Sur Mac, configurez plutôt LLM_BACKEND=mlx."
        )

    if os.getenv("HF_LOAD_IN_4BIT", "").lower() in {"1", "true", "yes"}:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            return _fail_preflight(
                "HF_LOAD_IN_4BIT=true mais bitsandbytes est absent.\n"
                'Installez le backend CUDA avec: pip install -e ".[dev,ui,cuda]"'
            )

    return None
