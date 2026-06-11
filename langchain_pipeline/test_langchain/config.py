"""Configuração do protótipo LangChain."""

import os

from common.config import BaseConfig


class Config(BaseConfig):
    """Herda configuração compartilhada e adiciona ajustes do LangChain.

    Guardrails e engenharia de contexto ficam ativos por padrão e podem ser
    desligados por variável de ambiente para fins de comparação no benchmark.
    """

    def __init__(self) -> None:
        super().__init__()
        self.guardrails_enabled = self._bool_env("LANGCHAIN_GUARDRAILS", True)
        self.context_engineering_enabled = self._bool_env(
            "LANGCHAIN_CONTEXT_ENGINEERING", True
        )
        self.context_max_chars = self._int_env("LANGCHAIN_CONTEXT_MAX_CHARS", 1500)
        self.notes_max_points = self._int_env("LANGCHAIN_NOTES_MAX_POINTS", 6)
        self.input_max_chars = self._int_env("LANGCHAIN_INPUT_MAX_CHARS", 2000)

    @staticmethod
    def _bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on", "sim"}

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        raw = os.getenv(name, str(default))
        try:
            return int(raw)
        except ValueError:
            return default
