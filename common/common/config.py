"""Configuração compartilhada entre frameworks."""

import os


class BaseConfig:
    """Variáveis de ambiente usadas por todos os agentes."""

    def __init__(self) -> None:
        self.openai_api_key = self._require_env("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_temperature = self._float_env("OPENAI_TEMPERATURE", 0.0)

    @staticmethod
    def _require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise ValueError(f"{name} é obrigatório")
        return value

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        raw = os.getenv(name, str(default))
        try:
            return float(raw)
        except ValueError:
            return default
