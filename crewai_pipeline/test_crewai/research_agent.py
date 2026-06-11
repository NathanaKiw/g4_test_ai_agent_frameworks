"""Agente de Pesquisa e Relatório usando CrewAI.

Implementa o mesmo caso de uso com três agentes especializados e três tarefas
executadas em processo sequencial.
"""

import time
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict
import logging

from common.common import (
    aggregate_token_usages,
    ResearchDataService,
    analyst_system,
    analyst_user,
    build_result,
    estimate_token_usage,
    extract_token_usage,
    report_writer_system,
    report_writer_user,
    researcher_system,
    researcher_user,
)
from common.common import get_logger

from .config import Config


def _error_code(exc: BaseException) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = error.get("code")
            return str(code) if code else None
    return None


def _is_insufficient_quota_error(exc: BaseException) -> bool:
    return _error_code(exc) == "insufficient_quota"


@contextmanager
def _suppress_crewai_logging():
    logger_names = (
        "crewai",
        "crewai.flow",
        "crewai.flow.flow",
        "CrewAIEventsBus",
    )
    previous_levels = {}
    try:
        for name in logger_names:
            logger = logging.getLogger(name)
            previous_levels[name] = logger.level
            logger.setLevel(logging.CRITICAL)
        yield
    finally:
        for name, previous_level in previous_levels.items():
            logging.getLogger(name).setLevel(previous_level)


class CrewAIResearchReportAgent:
    """Pipeline pesquisa → análise → relatório orquestrado por CrewAI."""

    FRAMEWORK = "CrewAI"

    def __init__(self) -> None:
        try:
            from crewai import Agent, Crew, LLM, Process, Task
        except ImportError as exc:
            raise ImportError(
                "CrewAI não está instalado. Rode `pip install -e ./crewai_pipeline` "
                "ou `pip install -r requirements.txt`."
            ) from exc

        self.Agent = Agent
        self.Crew = Crew
        self.LLM = LLM
        self.Process = Process
        self.Task = Task

        self.config = Config()
        self.logger = get_logger("crewai_research_agent")
        self.research_service = ResearchDataService()
        self._stage_llms: Dict[str, Any] = {}
        self._last_api_calls = 0

    def _build_agents(self):
        research_llm = self.LLM(
            model=self.config.groq_model,
            temperature=self.config.groq_temperature,
            api_key=self.config.groq_api_key,
            base_url=self.config.groq_base_url,
        )
        analysis_llm = self.LLM(
            model=self.config.groq_model,
            temperature=self.config.groq_temperature,
            api_key=self.config.groq_api_key,
            base_url=self.config.groq_base_url,
        )
        report_llm = self.LLM(
            model=self.config.groq_model,
            temperature=self.config.groq_temperature,
            api_key=self.config.groq_api_key,
            base_url=self.config.groq_base_url,
        )
        self._stage_llms = {
            "research": research_llm,
            "analysis": analysis_llm,
            "report": report_llm,
        }
        researcher = self.Agent(
            role="Pesquisador",
            goal="Coletar contexto, fatos, tendências e oportunidades sobre o tópico.",
            backstory=researcher_system(),
            llm=research_llm,
            verbose=False,
            allow_delegation=False,
        )
        analyst = self.Agent(
            role="Analista",
            goal="Interpretar a pesquisa e produzir insights acionáveis.",
            backstory=analyst_system(),
            llm=analysis_llm,
            verbose=False,
            allow_delegation=False,
        )
        writer = self.Agent(
            role="Redator Executivo",
            goal="Transformar a análise em um relatório executivo profissional.",
            backstory=report_writer_system(),
            llm=report_llm,
            verbose=False,
            allow_delegation=False,
        )
        return researcher, analyst, writer

    def _task_callback_factory(self, stage_timings: Dict[str, float]):
        stage_order = ["research_s", "analysis_s", "report_s"]
        last_checkpoint = {"time": time.perf_counter(), "index": 0}

        def callback(_output) -> None:
            index = last_checkpoint["index"]
            if index >= len(stage_order):
                return

            now = time.perf_counter()
            stage_timings[stage_order[index]] = round(
                now - last_checkpoint["time"], 2
            )
            last_checkpoint["time"] = now
            last_checkpoint["index"] = index + 1
            self._last_api_calls += 1

        return callback

    def _build_tasks(self, topic: str, stage_timings: Dict[str, float]):
        researcher, analyst, writer = self._build_agents()
        task_callback = self._task_callback_factory(stage_timings)
        self._stage_inputs = {
            "research": (researcher_system(), researcher_user(topic)),
            "analysis": (
                analyst_system(),
                analyst_user(topic, "Use como base o resultado da tarefa de pesquisa anterior."),
            ),
            "report": (
                report_writer_system(),
                report_writer_user(topic, "Use como base o resultado da tarefa de análise anterior."),
            ),
        }

        research_task = self.Task(
            description=researcher_user(topic),
            expected_output=(
                "Relatório de pesquisa estruturado com contexto, fatos, "
                "tendências, desafios, oportunidades e referências conceituais."
            ),
            agent=researcher,
            callback=task_callback,
        )
        analysis_task = self.Task(
            description=analyst_user(
                topic,
                "Use como base o resultado da tarefa de pesquisa anterior.",
            ),
            expected_output=(
                "Análise estruturada com tendências, riscos, cenários, lacunas "
                "e conclusões preliminares."
            ),
            agent=analyst,
            context=[research_task],
            callback=task_callback,
        )
        report_task = self.Task(
            description=report_writer_user(
                topic,
                "Use como base o resultado da tarefa de análise anterior.",
            ),
            expected_output=(
                "Relatório executivo final em português, com markdown leve, "
                "resumo executivo, achados, implicações e recomendações."
            ),
            agent=writer,
            context=[analysis_task],
            callback=task_callback,
        )
        return [research_task, analysis_task, report_task], [
            researcher,
            analyst,
            writer,
        ]

    @staticmethod
    def _raw_output(task) -> str:
        output = getattr(task, "output", None)
        raw = getattr(output, "raw", None)
        return raw if raw is not None else str(output or "")

    def _usage_for_stage(self, stage: str, output_text: str):
        llm = self._stage_llms.get(stage)
        usage = extract_token_usage(llm.get_token_usage_summary()) if llm is not None else None
        if not usage or not (usage.total_tokens or usage.prompt_tokens or usage.completion_tokens):
            prompt_text = "\n\n".join(self._stage_inputs.get(stage, ("", "")))
            usage = estimate_token_usage(prompt_text=prompt_text, completion_text=output_text)
        return usage

    def run_research_pipeline(self, topic: str) -> Dict[str, Any]:
        """Executa o crew sequencial para um tópico."""
        self._last_api_calls = 0
        self.logger.info("Iniciando pipeline CrewAI para: %s", topic)

        stage_timings: Dict[str, float] = {}
        tasks, agents = self._build_tasks(topic, stage_timings)
        crew = self.Crew(
            agents=agents,
            tasks=tasks,
            process=self.Process.sequential,
            verbose=False,
        )

        t0 = time.perf_counter()
        with _suppress_crewai_logging():
            try:
                crew.kickoff(inputs={"topic": topic})
            except Exception as exc:
                if _is_insufficient_quota_error(exc):
                    raise ValueError(
                        "Quota da API Groq esgotada para o CrewAI. "
                        "Atualize o GROQ_API_KEY, verifique billing/faturamento "
                        "ou use um modelo com quota disponível em GROQ_MODEL."
                    ) from exc
                raise
        self._last_api_calls = max(self._last_api_calls, len(tasks))

        stage_keys = ("research_s", "analysis_s", "report_s")
        if all(key in stage_timings for key in stage_keys):
            stage_timings["total_s"] = round(
                sum(stage_timings[key] for key in stage_keys), 2
            )
        else:
            stage_timings["total_s"] = round(time.perf_counter() - t0, 2)

        research_output = self._raw_output(tasks[0])
        analysis_output = self._raw_output(tasks[1])
        report_output = self._raw_output(tasks[2])

        research_usage = self._usage_for_stage("research", research_output)
        analysis_usage = self._usage_for_stage("analysis", analysis_output)
        report_usage = self._usage_for_stage("report", report_output)

        result = build_result(
            topic=topic,
            research=research_output,
            analysis=analysis_output,
            report=report_output,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        result["framework"] = self.FRAMEWORK
        result["api_calls"] = self._last_api_calls
        result["stage_timings"] = stage_timings
        result["stage_token_usage"] = {
            "research": research_usage.to_dict(),
            "analysis": analysis_usage.to_dict(),
            "report": report_usage.to_dict(),
        }
        result["token_usage"] = aggregate_token_usages(
            [research_usage, analysis_usage, report_usage]
        ).to_dict()

        self.research_service.insert_research_report(result)
        self.logger.info(
            "Pipeline CrewAI concluído (%d tarefas, total=%.2fs)",
            len(tasks),
            stage_timings["total_s"],
        )
        return result

    def get_api_calls_count(self) -> int:
        return self._last_api_calls

    def close(self) -> None:
        self.research_service.close()
