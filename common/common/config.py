"""Configuração compartilhada entre frameworks."""

import os


class BaseConfig:
    """Variáveis de ambiente usadas por todos os agentes."""

    def __init__(self) -> None:
        self.groq_api_key = self._require_env("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.groq_temperature = self._float_env("GROQ_TEMPERATURE", 0.0)
        self.groq_base_url = os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )

        self.openai_api_key = self.groq_api_key
        self.openai_model = self.groq_model
        self.openai_temperature = self.groq_temperature
        self.openai_base_url = self.groq_base_url

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
