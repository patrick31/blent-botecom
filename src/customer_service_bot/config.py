"""Configuration de l'assistant et représentation de l'utilisateur authentifié."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "infos" / "orders.db"


@dataclass(frozen=True)
class AuthenticatedUser:
    """Utilisateur déjà authentifié par l'application appelante."""

    user_id: int
    first_name: str
    last_name: str
    email: str

    @property
    def display_name(self) -> str:
        """Retourne un nom lisible pour personnaliser les réponses."""

        return f"{self.first_name} {self.last_name}".strip()


@dataclass(frozen=True)
class AppConfig:
    """Paramètres techniques utilisés par l'assistant."""

    database_path: Path = DEFAULT_DATABASE_PATH
    llm_model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    embedding_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_new_tokens: int = 180
    analysis_max_new_tokens: int = 160
    temperature: float = 0.0
    llm_backend: str = "transformers"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    load_in_4bit: bool = False
    use_llm: bool = True

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Construit la configuration depuis les variables d'environnement."""

        database_path = Path(os.getenv("ORDERS_DATABASE_PATH", DEFAULT_DATABASE_PATH))
        return cls(
            database_path=database_path,
            llm_model_id=os.getenv("HF_LLM_MODEL_ID", cls.llm_model_id),
            embedding_model_id=os.getenv("HF_EMBEDDING_MODEL_ID", cls.embedding_model_id),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", str(cls.max_new_tokens))),
            analysis_max_new_tokens=int(os.getenv("ANALYSIS_MAX_NEW_TOKENS", str(cls.analysis_max_new_tokens))),
            temperature=float(os.getenv("LLM_TEMPERATURE", str(cls.temperature))),
            llm_backend=os.getenv("LLM_BACKEND", cls.llm_backend),
            device_map=os.getenv("HF_DEVICE_MAP", cls.device_map),
            torch_dtype=os.getenv("HF_TORCH_DTYPE", cls.torch_dtype),
            load_in_4bit=os.getenv("HF_LOAD_IN_4BIT", "false").lower() in {"1", "true", "yes", "on"},
            use_llm=os.getenv("USE_LLM", "true").lower() in {"1", "true", "yes", "on"},
        )
