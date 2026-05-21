"""Agente de Pesquisa e Relatório — baseline vanilla (API OpenAI direta).

Sem frameworks: controle explícito do fluxo via três chamadas sequenciais
a chat.completions, servindo como linha de base do experimento.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict

from openai import OpenAI

from common.common import (
    ResearchDataService,
    analyst_messages,
    build_result,
    report_writer_messages,
    researcher_messages,
)
from common.common.logging_config import get_logger

from .config import Config

logger = get_logger("vanilla_research_agent")


class ResearchReportAgent:
    """Pipeline pesquisa → análise → relatório via API REST/SDK OpenAI."""

    FRAMEWORK = "Vanilla (OpenAI API)"
    API_CALLS_PER_RUN = 3

    def __init__(self) -> None:
        self.config = Config()
        self.logger = logging.getLogger("vanilla_research_agent")
        self.research_service = ResearchDataService()
        self.client = OpenAI(api_key=self.config.openai_api_key)
        self._last_api_calls = 0

    def _chat(self, system_content: str, user_content: str) -> str:
        """Uma chamada direta ao endpoint de chat completions."""
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

    def run_research_pipeline(self, topic: str) -> Dict[str, Any]:
        """Executa o fluxo de três etapas com controle manual total."""
        self._last_api_calls = 0
        self.logger.info("Iniciando pipeline vanilla para: %s", topic)
        stage_timings: Dict[str, float] = {}

        system_msg, user_msg = researcher_messages(topic)
        t0 = time.perf_counter()
        research = self._chat(system_msg, user_msg)
        stage_timings["research_s"] = time.perf_counter() - t0

        system_msg, user_msg = analyst_messages(topic, research)
        t0 = time.perf_counter()
        analysis = self._chat(system_msg, user_msg)
        stage_timings["analysis_s"] = time.perf_counter() - t0

        system_msg, user_msg = report_writer_messages(topic, analysis)
        t0 = time.perf_counter()
        report = self._chat(system_msg, user_msg)
        stage_timings["report_s"] = time.perf_counter() - t0

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
        self.logger.info("Pipeline vanilla concluído (%d chamadas API)", self._last_api_calls)
        return result

    def get_api_calls_count(self) -> int:
        return self._last_api_calls

    def close(self) -> None:
        self.research_service.close()
