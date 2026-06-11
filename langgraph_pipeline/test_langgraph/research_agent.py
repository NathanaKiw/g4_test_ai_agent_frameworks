"""Agente de Pesquisa e Relatório usando LangGraph.

Implementa o fluxo canônico (pesquisa → análise → redação) em um
``StateGraph`` e integra, de forma transversal:

- **Guardrails**: validação de entrada (nó ``input_guard``) e higienização
  da saída de cada etapa (redação de segredos).
- **Engenharia de contexto**: compactação do histórico repassado entre etapas
  e notas estruturadas acumuladas como memória de trabalho.

Guardrails e engenharia de contexto são determinísticos e não consomem
chamadas de API adicionais, preservando ``api_calls == 3``.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, TypedDict

from common.common import (
    aggregate_token_usages,
    ResearchDataService,
    StructuredNotes,
    analyst_messages,
    build_result,
    check_input,
    compact_history,
    estimate_token_usage,
    extract_structured_notes,
    extract_token_usage,
    GuardrailError,
    render_notes,
    report_writer_messages,
    researcher_messages,
    sanitize_output,
)
from common.common import get_logger
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
    stage_token_usage: Dict[str, Dict[str, Any]]
    notes: StructuredNotes
    compaction: Dict[str, Dict[str, Any]]
    guardrails: Dict[str, Any]


class LangGraphResearchReportAgent:
    """Pipeline pesquisa → análise → relatório orquestrado por LangGraph."""

    FRAMEWORK = "LangGraph"

    # Padrões usados quando ``self.config`` não está disponível (ex.: testes).
    _DEFAULTS = {
        "guardrails_enabled": True,
        "context_engineering_enabled": True,
        "context_max_chars": 1500,
        "notes_max_points": 6,
        "input_max_chars": 2000,
    }

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

    # ------------------------------------------------------------------
    # Configuração tolerante (funciona mesmo sem ``self.config``)
    # ------------------------------------------------------------------
    def _setting(self, name: str, default: Any = None) -> Any:
        if default is None:
            default = self._DEFAULTS.get(name)
        return getattr(getattr(self, "config", None), name, default)

    def _chat(self, system_content: str, user_content: str):
        self._last_api_calls += 1
        response = self.llm.invoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ]
        )
        content = response.content
        text = content if isinstance(content, str) else str(content)
        usage = extract_token_usage(response)
        if not usage.total_tokens:
            usage = estimate_token_usage(
                prompt_text=f"{system_content}\n{user_content}",
                completion_text=text,
            )
        return text, usage

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("input_guard", self._input_guard_node)
        graph.add_node("research", self._research_node)
        graph.add_node("analysis", self._analysis_node)
        graph.add_node("report", self._report_node)
        graph.add_edge(START, "input_guard")
        graph.add_edge("input_guard", "research")
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

    # ------------------------------------------------------------------
    # Guardrails (transversais)
    # ------------------------------------------------------------------
    def _input_guard_node(self, state: ResearchState) -> ResearchState:
        """Valida o tópico antes de gastar chamadas de API."""
        guardrails: Dict[str, Any] = {
            "enabled": self._setting("guardrails_enabled"),
            "input": {},
            "output_redactions": 0,
            "output_violations": [],
        }
        if not self._setting("guardrails_enabled"):
            return {"guardrails": guardrails}

        result = check_input(
            state["topic"], max_chars=self._setting("input_max_chars")
        )
        guardrails["input"] = result.to_dict()
        if not result.allowed:
            self.logger.warning(
                "Entrada bloqueada pelos guardrails: %s", result.violations
            )
            raise GuardrailError(
                "Tópico rejeitado pelos guardrails de entrada: "
                + ", ".join(result.violations)
            )
        return {"guardrails": guardrails}

    def _sanitize_output(
        self, state: ResearchState, text: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Higieniza a saída de uma etapa e atualiza o estado de guardrails."""
        guardrails = dict(state.get("guardrails", {}))
        if not self._setting("guardrails_enabled"):
            return text, guardrails

        result = sanitize_output(text)
        guardrails["output_redactions"] = (
            guardrails.get("output_redactions", 0) + result.redactions
        )
        if result.violations:
            merged = guardrails.get("output_violations", []) + result.violations
            guardrails["output_violations"] = list(dict.fromkeys(merged))
        return result.text, guardrails

    # ------------------------------------------------------------------
    # Engenharia de contexto (transversal)
    # ------------------------------------------------------------------
    def _record_notes(
        self, state: ResearchState, stage: str, text: str
    ) -> StructuredNotes:
        """Acrescenta as notas estruturadas extraídas de uma etapa."""
        notes = state.get("notes") or StructuredNotes()
        if self._setting("context_engineering_enabled"):
            points = extract_structured_notes(
                stage, text, max_points=self._setting("notes_max_points")
            )
            notes.add(stage, points)
        return notes

    def _prepare_context(
        self, state: ResearchState, raw_text: str, slot: str
    ) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """Compacta o histórico e injeta as notas para a próxima etapa.

        Returns:
            Par ``(texto_para_o_prompt, compaction_atualizado)``.
        """
        compaction = dict(state.get("compaction", {}))
        if not self._setting("context_engineering_enabled"):
            return raw_text, compaction

        result = compact_history(
            raw_text, max_chars=self._setting("context_max_chars")
        )
        compaction[slot] = result.to_dict()

        notes_block = render_notes(state.get("notes") or StructuredNotes())
        augmented = (
            f"{notes_block}\n\n{result.text}" if notes_block else result.text
        )
        return augmented, compaction

    # ------------------------------------------------------------------
    # Nós do pipeline
    # ------------------------------------------------------------------
    def _research_node(self, state: ResearchState) -> ResearchState:
        system_msg, user_msg = researcher_messages(state["topic"])
        t0 = time.perf_counter()
        research, usage = self._chat(system_msg, user_msg)
        research, guardrails = self._sanitize_output(state, research)
        notes = self._record_notes(state, "pesquisa", research)
        return {
            "research": research,
            "notes": notes,
            "guardrails": guardrails,
            "stage_timings": self._record_timing(state, "research_s", t0),
            "stage_token_usage": {
                **state.get("stage_token_usage", {}),
                "research": usage.to_dict(),
            },
        }

    def _analysis_node(self, state: ResearchState) -> ResearchState:
        research_context, compaction = self._prepare_context(
            state, state["research"], "analysis_input"
        )
        system_msg, user_msg = analyst_messages(state["topic"], research_context)
        t0 = time.perf_counter()
        analysis, usage = self._chat(system_msg, user_msg)
        analysis, guardrails = self._sanitize_output(state, analysis)
        notes = self._record_notes(state, "análise", analysis)
        return {
            "analysis": analysis,
            "notes": notes,
            "compaction": compaction,
            "guardrails": guardrails,
            "stage_timings": self._record_timing(state, "analysis_s", t0),
            "stage_token_usage": {
                **state.get("stage_token_usage", {}),
                "analysis": usage.to_dict(),
            },
        }

    def _report_node(self, state: ResearchState) -> ResearchState:
        analysis_context, compaction = self._prepare_context(
            state, state["analysis"], "report_input"
        )
        system_msg, user_msg = report_writer_messages(
            state["topic"], analysis_context
        )
        t0 = time.perf_counter()
        report, usage = self._chat(system_msg, user_msg)
        report, guardrails = self._sanitize_output(state, report)
        return {
            "report": report,
            "compaction": compaction,
            "guardrails": guardrails,
            "stage_timings": self._record_timing(state, "report_s", t0),
            "stage_token_usage": {
                **state.get("stage_token_usage", {}),
                "report": usage.to_dict(),
            },
        }

    def _build_context_summary(self, final_state: ResearchState) -> Dict[str, Any]:
        """Consolida as métricas de engenharia de contexto para o resultado."""
        compaction = dict(final_state.get("compaction", {}))
        notes = final_state.get("notes") or StructuredNotes()
        chars_saved = sum(
            max(0, entry.get("original_chars", 0) - entry.get("compacted_chars", 0))
            for entry in compaction.values()
        )
        return {
            "enabled": self._setting("context_engineering_enabled"),
            "notes": notes.to_dict(),
            "compaction": compaction,
            "chars_saved": chars_saved,
        }

    def run_research_pipeline(self, topic: str) -> Dict[str, Any]:
        """Executa o grafo completo para um tópico."""
        self._last_api_calls = 0
        self.logger.info("Iniciando pipeline LangGraph para: %s", topic)

        final_state = self.graph.invoke({"topic": topic, "stage_timings": {}})
        stage_timings = dict(final_state.get("stage_timings", {}))
        stage_timings["total_s"] = round(sum(stage_timings.values()), 2)
        stage_token_usage = dict(final_state.get("stage_token_usage", {}))
        token_usage = aggregate_token_usages(
            [
                extract_token_usage(stage_token_usage.get("research")),
                extract_token_usage(stage_token_usage.get("analysis")),
                extract_token_usage(stage_token_usage.get("report")),
            ]
        )

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
        result["token_usage"] = token_usage.to_dict()
        result["stage_token_usage"] = stage_token_usage
        result["guardrails"] = final_state.get("guardrails", {})
        result["context_engineering"] = self._build_context_summary(final_state)

        self.research_service.insert_research_report(result)
        self.logger.info(
            "Pipeline LangGraph concluído (%d chamadas API, total=%.2fs, "
            "%d caracteres economizados por compactação)",
            self._last_api_calls,
            stage_timings["total_s"],
            result["context_engineering"]["chars_saved"],
        )
        return result

    def get_api_calls_count(self) -> int:
        return self._last_api_calls

    def close(self) -> None:
        self.research_service.close()
