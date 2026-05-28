"""Agente de Pesquisa e Relatório — baseline vanilla (API OpenAI direta).

Sem frameworks: controle explícito do fluxo via três chamadas sequenciais
a chat.completions, servindo como linha de base do experimento.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from common import (
    ResearchDataService,
    analyst_messages,
    build_result,
    report_writer_messages,
    researcher_messages,
)
from common.logging_config import get_logger

from .config import Config

logger = get_logger("vanilla_research_agent")

def _is_retryable_status_error(exc: BaseException) -> bool:
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def _error_code(exc: BaseException) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = error.get("code")
            return str(code) if code else None
    return None


def _is_retryable_openai_error(exc: BaseException) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, RateLimitError):
        return _error_code(exc) != "insufficient_quota"
    return _is_retryable_status_error(exc)


class ResearchReportAgent:
    """Pipeline pesquisa → análise → relatório via API REST/SDK OpenAI."""

    FRAMEWORK = "Vanilla (OpenAI API)"
    API_CALLS_PER_RUN = 3

    def __init__(self) -> None:
        self.config = Config()
        self.logger = get_logger("vanilla_research_agent")
        self.research_service = ResearchDataService()
        self.client = OpenAI(api_key=self.config.openai_api_key)
        self._last_api_calls = 0

    def _chat(self, system_content: str, user_content: str) -> str:
        """Uma chamada direta ao endpoint de chat completions com retry automático.

        Realiza até 3 tentativas com espera exponencial em caso de erros
        transitórios (rate limit, falha de conexão). Erros definitivos
        (ex.: chave inválida, modelo inexistente) são propagados imediatamente.
        """

        @retry(
            retry=retry_if_exception(_is_retryable_openai_error),
            wait=wait_exponential(multiplier=1, min=4, max=30),
            stop=stop_after_attempt(3),
            before_sleep=before_sleep_log(self.logger, logging.WARNING),
            reraise=True,
        )
        def _call() -> str:
            self._last_api_calls += 1
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                temperature=self.config.openai_temperature,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            )
            return response.choices[0].message.content or ""

        return _call()

    def run_research_pipeline(self, topic: str) -> Dict[str, Any]:
        """Executa o fluxo de três etapas com controle manual total."""
        self._last_api_calls = 0
        self.logger.info("Iniciando pipeline vanilla para: %s", topic)
        stage_timings: Dict[str, float] = {}

        # Etapa 1 — Pesquisador
        system_msg, user_msg = researcher_messages(topic)
        t0 = time.perf_counter()
        research = self._chat(system_msg, user_msg)
        stage_timings["research_s"] = round(time.perf_counter() - t0, 2)

        # Etapa 2 — Analista
        system_msg, user_msg = analyst_messages(topic, research)
        t0 = time.perf_counter()
        analysis = self._chat(system_msg, user_msg)
        stage_timings["analysis_s"] = round(time.perf_counter() - t0, 2)

        # Etapa 3 — Redator
        system_msg, user_msg = report_writer_messages(topic, analysis)
        t0 = time.perf_counter()
        report = self._chat(system_msg, user_msg)
        stage_timings["report_s"] = round(time.perf_counter() - t0, 2)

        stage_timings["total_s"] = round(
            stage_timings["research_s"]
            + stage_timings["analysis_s"]
            + stage_timings["report_s"],
            2,
        )

        result = build_result(
            topic=topic,
            research=research,
            analysis=analysis,
            report=report,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        result["framework"] = self.FRAMEWORK
        result["api_calls"] = self._last_api_calls
        result["stage_timings"] = stage_timings

        self.research_service.insert_research_report(result)
        self.logger.info(
            "Pipeline vanilla concluído (%d chamadas API, total=%.2fs)",
            self._last_api_calls,
            stage_timings["total_s"],
        )
        return result

    def get_api_calls_count(self) -> int:
        return self._last_api_calls

    def close(self) -> None:
        self.research_service.close()
