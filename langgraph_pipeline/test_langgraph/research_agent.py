"""Agente de Pesquisa e Relatório usando LangGraph.

Implementa o mesmo fluxo canônico do baseline vanilla com três nós
explícitos em um StateGraph: pesquisa, análise e redação.
"""

import time
from datetime import datetime
from typing import Any, Dict, TypedDict

from common import (
    ResearchDataService,
    analyst_messages,
    build_result,
    report_writer_messages,
    researcher_messages,
)
from common.logging_config import get_logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .config import Config


class ResearchState(TypedDict, total=False):
    topic: str
    research: str
    analysis: str
    report: str
    stage_timings: Dict[str, float]


class LangGraphResearchReportAgent:
    """Pipeline pesquisa → análise → relatório orquestrado por LangGraph."""

    FRAMEWORK = "LangGraph"

    def __init__(self) -> None:
        self.config = Config()
        self.logger = get_logger("langgraph_research_agent")
        self.research_service = ResearchDataService()
        self.llm = ChatOpenAI(
            model=self.config.groq_model,
            temperature=self.config.groq_temperature,
            api_key=self.config.groq_api_key,
            base_url=self.config.groq_base_url,
        )
        self._last_api_calls = 0
        self.graph = self._build_graph()

    def _chat(self, system_content: str, user_content: str) -> str:
        self._last_api_calls += 1
        response = self.llm.invoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ]
        )
        content = response.content
        return content if isinstance(content, str) else str(content)

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("research", self._research_node)
        graph.add_node("analysis", self._analysis_node)
        graph.add_node("report", self._report_node)
        graph.add_edge(START, "research")
        graph.add_edge("research", "analysis")
        graph.add_edge("analysis", "report")
        graph.add_edge("report", END)
        return graph.compile()

    def _record_timing(
        self, state: ResearchState, key: str, start_time: float
    ) -> Dict[str, float]:
        timings = dict(state.get("stage_timings", {}))
        timings[key] = round(time.perf_counter() - start_time, 2)
        return timings

    def _research_node(self, state: ResearchState) -> ResearchState:
        system_msg, user_msg = researcher_messages(state["topic"])
        t0 = time.perf_counter()
        research = self._chat(system_msg, user_msg)
        return {
            "research": research,
            "stage_timings": self._record_timing(state, "research_s", t0),
        }

    def _analysis_node(self, state: ResearchState) -> ResearchState:
        system_msg, user_msg = analyst_messages(state["topic"], state["research"])
        t0 = time.perf_counter()
        analysis = self._chat(system_msg, user_msg)
        return {
            "analysis": analysis,
            "stage_timings": self._record_timing(state, "analysis_s", t0),
        }

    def _report_node(self, state: ResearchState) -> ResearchState:
        system_msg, user_msg = report_writer_messages(state["topic"], state["analysis"])
        t0 = time.perf_counter()
        report = self._chat(system_msg, user_msg)
        return {
            "report": report,
            "stage_timings": self._record_timing(state, "report_s", t0),
        }

    def run_research_pipeline(self, topic: str) -> Dict[str, Any]:
        """Executa o grafo completo para um tópico."""
        self._last_api_calls = 0
        self.logger.info("Iniciando pipeline LangGraph para: %s", topic)

        final_state = self.graph.invoke({"topic": topic, "stage_timings": {}})
        stage_timings = dict(final_state.get("stage_timings", {}))
        stage_timings["total_s"] = round(sum(stage_timings.values()), 2)

        result = build_result(
            topic=topic,
            research=final_state["research"],
            analysis=final_state["analysis"],
            report=final_state["report"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        result["framework"] = self.FRAMEWORK
        result["api_calls"] = self._last_api_calls
        result["stage_timings"] = stage_timings

        self.research_service.insert_research_report(result)
        self.logger.info(
            "Pipeline LangGraph concluído (%d chamadas API, total=%.2fs)",
            self._last_api_calls,
            stage_timings["total_s"],
        )
        return result

    def get_api_calls_count(self) -> int:
        return self._last_api_calls

    def close(self) -> None:
        self.research_service.close()
