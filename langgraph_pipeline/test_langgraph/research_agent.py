"""Agente de Pesquisa e Relatório usando LangGraph.

Implementa o fluxo canônico (pesquisa → análise → redação) em um
``StateGraph`` e integra, de forma transversal:

- **Guardrails**: validação de entrada (nó ``input_guard``) e higienização
  da saída de cada etapa (redação de segredos).
- **Engenharia de contexto**: compactação do histórico repassado entre etapas
  e notas estruturadas acumuladas como memória de trabalho.
- **Estados duráveis** (opt-in via ``durable=True``): checkpointing do estado a
  cada nó, com ``thread_id`` para consultar o estado persistido e **retomar**
  uma execução interrompida sem reprocessar etapas concluídas.

Guardrails e engenharia de contexto são determinísticos e não consomem
chamadas de API adicionais, preservando ``api_calls == 3``. O modo durável é
opcional e desligado por padrão, mantendo o pipeline de benchmark intacto.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from uuid import uuid4

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
    notes: Dict[str, Any]
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

    # Padrões de classe — herdados por stubs que ignoram ``__init__``,
    # garantindo que o caminho de benchmark (não durável) permaneça intacto.
    durable: bool = False
    _checkpointer: Any = None
    _checkpointer_name: Optional[str] = None
    _last_thread_id: Optional[str] = None

    def __init__(self, *, durable: bool = False, checkpoint_db: Optional[str] = None) -> None:
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
        self.durable = durable
        self._checkpoint_db = checkpoint_db
        self._checkpointer = self._make_checkpointer() if durable else None
        self._checkpointer_name = (
            type(self._checkpointer).__name__ if self._checkpointer else None
        )
        self._last_thread_id = None
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Estados duráveis (checkpointing) — opt-in
    # ------------------------------------------------------------------
    def _make_checkpointer(self):
        """Cria o checkpointer do LangGraph.

        Prefere o ``SqliteSaver`` (durável entre processos) quando um caminho de
        banco é fornecido e o pacote está disponível; caso contrário, recorre ao
        ``MemorySaver`` (durável dentro do processo, suporta retomada).
        """
        if getattr(self, "_checkpoint_db", None):
            try:
                import sqlite3

                from langgraph.checkpoint.sqlite import SqliteSaver

                conn = sqlite3.connect(self._checkpoint_db, check_same_thread=False)
                return SqliteSaver(conn)
            except Exception as exc:  # pragma: no cover - depende de pacote opcional
                self.logger.warning(
                    "SqliteSaver indisponível (%s); usando MemorySaver. Para "
                    "durabilidade entre processos: pip install "
                    "langgraph-checkpoint-sqlite",
                    exc,
                )

        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    @staticmethod
    def _thread_config(thread_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _count_checkpoints(self, thread_id: str) -> int:
        try:
            return sum(1 for _ in self.graph.get_state_history(self._thread_config(thread_id)))
        except Exception:  # pragma: no cover - defensivo
            return 0

    def get_durable_state(self, thread_id: str) -> Dict[str, Any]:
        """Lê o estado persistido de uma ``thread`` (prova de durabilidade)."""
        if not getattr(self, "durable", False):
            raise RuntimeError("Modo durável desativado: instancie com durable=True.")
        snapshot = self.graph.get_state(self._thread_config(thread_id))
        return dict(snapshot.values) if snapshot and snapshot.values else {}

    def state_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Histórico de checkpoints da ``thread`` (mais recente primeiro)."""
        if not getattr(self, "durable", False):
            raise RuntimeError("Modo durável desativado: instancie com durable=True.")
        return [snap.values for snap in self.graph.get_state_history(self._thread_config(thread_id))]

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
        checkpointer = getattr(self, "_checkpointer", None)
        if checkpointer is not None:
            return graph.compile(checkpointer=checkpointer)
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
    ) -> Dict[str, Any]:
        """Acrescenta as notas estruturadas extraídas de uma etapa.

        Retorna um dicionário simples (serializável por checkpointers duráveis),
        evitando persistir o dataclass diretamente no estado.
        """
        notes = StructuredNotes.from_dict(state.get("notes"))
        if self._setting("context_engineering_enabled"):
            points = extract_structured_notes(
                stage, text, max_points=self._setting("notes_max_points")
            )
            notes.add(stage, points)
        return notes.to_dict()

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

        notes_block = render_notes(StructuredNotes.from_dict(state.get("notes")))
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
        notes = StructuredNotes.from_dict(final_state.get("notes"))
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

    def _assemble_result(self, topic: str, final_state: ResearchState) -> Dict[str, Any]:
        """Monta o dicionário de resultado padronizado a partir do estado final."""
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
            research=final_state.get("research", ""),
            analysis=final_state.get("analysis", ""),
            report=final_state.get("report", ""),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        result["framework"] = self.FRAMEWORK
        result["api_calls"] = self._last_api_calls
        result["stage_timings"] = stage_timings
        result["token_usage"] = token_usage.to_dict()
        result["stage_token_usage"] = stage_token_usage
        result["guardrails"] = final_state.get("guardrails", {})
        result["context_engineering"] = self._build_context_summary(final_state)

        if getattr(self, "durable", False):
            thread_id = getattr(self, "_last_thread_id", None)
            result["durable"] = {
                "enabled": True,
                "thread_id": thread_id,
                "checkpointer": getattr(self, "_checkpointer_name", None),
                "checkpoints": self._count_checkpoints(thread_id) if thread_id else 0,
            }
        return result

    def run_research_pipeline(
        self, topic: str, *, thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executa o grafo completo para um tópico.

        No modo durável, o estado é persistido a cada nó sob ``thread_id``
        (gerado automaticamente se não informado), permitindo consulta e
        retomada posterior via :meth:`resume_research_pipeline`.
        """
        self._last_api_calls = 0
        self.logger.info("Iniciando pipeline LangGraph para: %s", topic)

        initial: ResearchState = {"topic": topic, "stage_timings": {}}
        if getattr(self, "durable", False):
            thread_id = thread_id or uuid4().hex
            self._last_thread_id = thread_id
            final_state = self.graph.invoke(initial, self._thread_config(thread_id))
        else:
            final_state = self.graph.invoke(initial)

        result = self._assemble_result(topic, final_state)
        self.research_service.insert_research_report(result)
        self.logger.info(
            "Pipeline LangGraph concluído (%d chamadas API, total=%.2fs, "
            "%d caracteres economizados por compactação)",
            self._last_api_calls,
            result["stage_timings"]["total_s"],
            result["context_engineering"]["chars_saved"],
        )
        return result

    def resume_research_pipeline(self, thread_id: str) -> Dict[str, Any]:
        """Retoma uma execução durável interrompida a partir do último checkpoint.

        Etapas já concluídas não são reprocessadas — seus resultados vêm do
        estado persistido —, demonstrando a durabilidade do estado.
        """
        if not getattr(self, "durable", False):
            raise RuntimeError("Modo durável desativado: instancie com durable=True.")

        self._last_thread_id = thread_id
        self.logger.info("Retomando pipeline LangGraph (thread=%s)", thread_id)
        final_state = self.graph.invoke(None, self._thread_config(thread_id))
        topic = final_state.get("topic", "")
        result = self._assemble_result(topic, final_state)
        self.research_service.insert_research_report(result)
        self.logger.info(
            "Pipeline LangGraph retomado e concluído (thread=%s)", thread_id
        )
        return result

    def get_api_calls_count(self) -> int:
        return self._last_api_calls

    def close(self) -> None:
        self.research_service.close()
