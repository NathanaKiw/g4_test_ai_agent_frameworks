"""Agente de Pesquisa e Relatório usando LangChain (LCEL).

Implementa o fluxo canônico (pesquisa → análise → redação) encadeando
componentes com a LangChain Expression Language: cada etapa é uma cadeia
``ChatPromptTemplate | ChatOpenAI`` cujo resultado é convertido por um
``StrOutputParser``.

Integra, de forma transversal (Semana 3):

- **Guardrails**: validação de entrada (``check_input``) antes de gastar
  chamadas de API e higienização da saída de cada etapa (``sanitize_output``).
- **Engenharia de contexto**: compactação do histórico (``compact_history``)
  repassado entre etapas e notas estruturadas (``extract_structured_notes`` /
  ``render_notes``) acumuladas como memória de trabalho.

Guardrails e engenharia de contexto são determinísticos e não consomem
chamadas de API adicionais, preservando ``api_calls == 3`` e mantendo a
comparação de benchmark justa entre os frameworks.
"""

import time
from datetime import datetime
from typing import Any, Dict, Tuple

from common.common import (
    GuardrailError,
    ResearchDataService,
    StructuredNotes,
    aggregate_token_usages,
    analyst_messages,
    build_result,
    check_input,
    compact_history,
    estimate_token_usage,
    extract_structured_notes,
    extract_token_usage,
    get_logger,
    render_notes,
    report_writer_messages,
    researcher_messages,
    sanitize_output,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Config


class LangChainResearchReportAgent:
    """Pipeline pesquisa → análise → relatório encadeado com LCEL."""

    FRAMEWORK = "LangChain"

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
        self.logger = get_logger("langchain_research_agent")
        self.research_service = ResearchDataService()
        self.llm = ChatOpenAI(
            model=self.config.groq_model,
            temperature=self.config.groq_temperature,
            api_key=self.config.groq_api_key,
            base_url=self.config.groq_base_url,
        )
        # Componentes LCEL reutilizados por todas as etapas. O conteúdo dinâmico
        # entra como VALOR das variáveis (não é re-interpretado como template),
        # o que evita problemas com chaves "{}" presentes no texto.
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_instructions}"),
                ("human", "{user_task}"),
            ]
        )
        self._parser = StrOutputParser()
        self._last_api_calls = 0

    # ------------------------------------------------------------------
    # Configuração tolerante (funciona mesmo sem ``self.config``)
    # ------------------------------------------------------------------
    def _setting(self, name: str, default: Any = None) -> Any:
        if default is None:
            default = self._DEFAULTS.get(name)
        return getattr(getattr(self, "config", None), name, default)

    def _chat(self, system_content: str, user_content: str):
        """Executa uma etapa como cadeia LCEL ``prompt | llm`` e a converte com
        ``StrOutputParser``, capturando o uso de tokens da resposta nativa."""
        self._last_api_calls += 1
        chain = self._prompt | self.llm
        message = chain.invoke(
            {"system_instructions": system_content, "user_task": user_content}
        )
        text = self._parser.invoke(message)
        usage = extract_token_usage(message)
        if not usage.total_tokens:
            usage = estimate_token_usage(
                prompt_text=f"{system_content}\n{user_content}",
                completion_text=text,
            )
        return text, usage

    # ------------------------------------------------------------------
    # Guardrails (transversais)
    # ------------------------------------------------------------------
    def _input_guard(self, topic: str) -> Dict[str, Any]:
        """Valida o tópico antes de iniciar o pipeline."""
        guardrails: Dict[str, Any] = {
            "enabled": self._setting("guardrails_enabled"),
            "input": {},
            "output_redactions": 0,
            "output_violations": [],
        }
        if not self._setting("guardrails_enabled"):
            return guardrails

        result = check_input(topic, max_chars=self._setting("input_max_chars"))
        guardrails["input"] = result.to_dict()
        if not result.allowed:
            self.logger.warning(
                "Entrada bloqueada pelos guardrails: %s", result.violations
            )
            raise GuardrailError(
                "Tópico rejeitado pelos guardrails de entrada: "
                + ", ".join(result.violations)
            )
        return guardrails

    def _sanitize_output(
        self, text: str, guardrails: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Higieniza a saída de uma etapa e atualiza o estado de guardrails."""
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
    def _record_notes(self, notes: StructuredNotes, stage: str, text: str) -> None:
        """Acrescenta as notas estruturadas extraídas de uma etapa."""
        if self._setting("context_engineering_enabled"):
            points = extract_structured_notes(
                stage, text, max_points=self._setting("notes_max_points")
            )
            notes.add(stage, points)

    def _prepare_context(
        self,
        raw_text: str,
        notes: StructuredNotes,
        slot: str,
        compaction: Dict[str, Dict[str, Any]],
    ) -> str:
        """Compacta o histórico e injeta as notas para a próxima etapa."""
        if not self._setting("context_engineering_enabled"):
            return raw_text

        result = compact_history(raw_text, max_chars=self._setting("context_max_chars"))
        compaction[slot] = result.to_dict()

        notes_block = render_notes(notes)
        return f"{notes_block}\n\n{result.text}" if notes_block else result.text

    def _build_context_summary(
        self, notes: StructuredNotes, compaction: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
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

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def run_research_pipeline(self, topic: str) -> Dict[str, Any]:
        """Executa o fluxo de três etapas encadeadas com LCEL."""
        self._last_api_calls = 0
        self.logger.info("Iniciando pipeline LangChain para: %s", topic)

        guardrails = self._input_guard(topic)
        notes = StructuredNotes()
        compaction: Dict[str, Dict[str, Any]] = {}
        stage_timings: Dict[str, float] = {}

        # Etapa 1 — Pesquisador
        system_msg, user_msg = researcher_messages(topic)
        t0 = time.perf_counter()
        research, research_usage = self._chat(system_msg, user_msg)
        research, guardrails = self._sanitize_output(research, guardrails)
        self._record_notes(notes, "pesquisa", research)
        stage_timings["research_s"] = round(time.perf_counter() - t0, 2)

        # Etapa 2 — Analista (recebe contexto compactado + notas)
        research_context = self._prepare_context(
            research, notes, "analysis_input", compaction
        )
        system_msg, user_msg = analyst_messages(topic, research_context)
        t0 = time.perf_counter()
        analysis, analysis_usage = self._chat(system_msg, user_msg)
        analysis, guardrails = self._sanitize_output(analysis, guardrails)
        self._record_notes(notes, "análise", analysis)
        stage_timings["analysis_s"] = round(time.perf_counter() - t0, 2)

        # Etapa 3 — Redator (recebe contexto compactado + notas)
        analysis_context = self._prepare_context(
            analysis, notes, "report_input", compaction
        )
        system_msg, user_msg = report_writer_messages(topic, analysis_context)
        t0 = time.perf_counter()
        report, report_usage = self._chat(system_msg, user_msg)
        report, guardrails = self._sanitize_output(report, guardrails)
        stage_timings["report_s"] = round(time.perf_counter() - t0, 2)

        stage_timings["total_s"] = round(
            stage_timings["research_s"]
            + stage_timings["analysis_s"]
            + stage_timings["report_s"],
            2,
        )

        stage_token_usage = {
            "research": research_usage.to_dict(),
            "analysis": analysis_usage.to_dict(),
            "report": report_usage.to_dict(),
        }
        token_usage = aggregate_token_usages(
            [research_usage, analysis_usage, report_usage]
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
        result["token_usage"] = token_usage.to_dict()
        result["stage_token_usage"] = stage_token_usage
        result["guardrails"] = guardrails
        result["context_engineering"] = self._build_context_summary(notes, compaction)

        self.research_service.insert_research_report(result)
        self.logger.info(
            "Pipeline LangChain concluído (%d chamadas API, total=%.2fs, "
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
