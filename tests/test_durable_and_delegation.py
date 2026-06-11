"""Testes dos modos opt-in: estado durável (LangGraph) e delegação (CrewAI).

Validam que:
- o LangGraph persiste o estado a cada nó (checkpointing), expõe o estado de
  uma ``thread`` e **retoma** uma execução interrompida sem reprocessar etapas
  já concluídas;
- o CrewAI, em modo hierárquico, monta o crew com ``Process.hierarchical``,
  um ``manager_llm`` e agentes com ``allow_delegation=True``;
- o caminho padrão (não durável / sequencial), usado pelo benchmark, segue
  inalterado.
"""

import unittest

from common.common import TokenUsage
from langgraph.checkpoint.memory import MemorySaver

from test_crewai.research_agent import CrewAIResearchReportAgent
from test_langgraph.research_agent import LangGraphResearchReportAgent


class _Recorder:
    def insert_research_report(self, report_data):
        return None

    def close(self):
        return None


class _SilentLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


# ----------------------------------------------------------------------
# LangGraph — estado durável
# ----------------------------------------------------------------------
class StubDurableLangGraph(LangGraphResearchReportAgent):
    """LangGraph durável com LLM determinístico e falha simulada opcional."""

    def __init__(self, fail_analysis_once: bool = False):
        self._last_api_calls = 0
        self.research_service = _Recorder()
        self.logger = _SilentLogger()
        self.durable = True
        self._checkpoint_db = None
        self._checkpointer = MemorySaver()
        self._checkpointer_name = "MemorySaver"
        self._last_thread_id = None
        self._fail_analysis_once = fail_analysis_once
        self.research_runs = 0
        self.graph = self._build_graph()

    def _chat(self, system_content, user_content):
        self._last_api_calls += 1
        text = "## Achados\n- ponto com 10% de crescimento\nConclusão: relevante."
        usage = TokenUsage(
            prompt_tokens=5, completion_tokens=9, total_tokens=14, source="actual"
        )
        return text, usage

    def _research_node(self, state):
        self.research_runs += 1
        return super()._research_node(state)

    def _analysis_node(self, state):
        if self._fail_analysis_once:
            self._fail_analysis_once = False
            raise RuntimeError("falha simulada na análise")
        return super()._analysis_node(state)


class DurableLangGraphTest(unittest.TestCase):
    def test_full_durable_run_exposes_thread_and_checkpoints(self):
        agent = StubDurableLangGraph()
        result = agent.run_research_pipeline("tema válido", thread_id="t-full")

        self.assertEqual(result["api_calls"], 3)
        self.assertTrue(result["durable"]["enabled"])
        self.assertEqual(result["durable"]["thread_id"], "t-full")
        self.assertEqual(result["durable"]["checkpointer"], "MemorySaver")
        self.assertGreater(result["durable"]["checkpoints"], 0)

        state = agent.get_durable_state("t-full")
        self.assertIn("research", state)
        self.assertIn("report", state)

    def test_state_survives_failure_and_resumes_without_recompute(self):
        agent = StubDurableLangGraph(fail_analysis_once=True)

        with self.assertRaises(RuntimeError):
            agent.run_research_pipeline("tema válido", thread_id="t-resume")

        # O estado da pesquisa foi persistido apesar da falha na análise.
        partial = agent.get_durable_state("t-resume")
        self.assertIn("research", partial)
        self.assertNotIn("analysis", partial)
        self.assertEqual(agent.research_runs, 1)

        # Retomada conclui o pipeline sem reprocessar a pesquisa.
        result = agent.resume_research_pipeline("t-resume")
        self.assertTrue(result["report"])
        self.assertEqual(agent.research_runs, 1)  # pesquisa NÃO foi refeita
        self.assertEqual(result["api_calls"], 3)  # 1 (antes) + 2 (retomada)

    def test_resume_requires_durable_mode(self):
        agent = StubDurableLangGraph()
        agent.durable = False
        with self.assertRaises(RuntimeError):
            agent.resume_research_pipeline("qualquer")


# ----------------------------------------------------------------------
# CrewAI — delegação autônoma (hierárquico)
# ----------------------------------------------------------------------
class _FakeOutput:
    def __init__(self, raw):
        self.raw = raw


class _FakeTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.output = _FakeOutput("conteúdo stub")


class _FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_token_usage_summary(self):
        return {}


class _FakeProcess:
    sequential = "sequential"
    hierarchical = "hierarchical"


class _FakeCrew:
    last_kwargs = None

    def __init__(self, **kwargs):
        _FakeCrew.last_kwargs = kwargs

    def kickoff(self, inputs=None):
        return None


class StubCrewAIAgent(CrewAIResearchReportAgent):
    """CrewAI com classes do framework substituídas por fakes determinísticos."""

    def __init__(self, hierarchical: bool = False):
        self._last_api_calls = 0
        self.research_service = _Recorder()
        self.logger = _SilentLogger()
        self._stage_llms = {}
        self.hierarchical = hierarchical
        self.Agent = _FakeAgent
        self.Crew = _FakeCrew
        self.LLM = _FakeLLM
        self.Process = _FakeProcess
        self.Task = _FakeTask

        class _Cfg:
            groq_model = "m"
            groq_temperature = 0.0
            groq_api_key = "k"
            groq_base_url = "u"

        self.config = _Cfg()


class DelegationCrewAITest(unittest.TestCase):
    def test_sequential_is_default_without_delegation(self):
        _FakeCrew.last_kwargs = None
        result = StubCrewAIAgent(hierarchical=False).run_research_pipeline("tema")

        self.assertEqual(_FakeCrew.last_kwargs["process"], _FakeProcess.sequential)
        self.assertNotIn("manager_llm", _FakeCrew.last_kwargs)
        self.assertFalse(result["delegation"]["enabled"])
        self.assertEqual(result["delegation"]["process"], "sequential")
        # Agentes sem delegação no modo sequencial.
        for agent in _FakeCrew.last_kwargs["agents"]:
            self.assertFalse(agent.kwargs.get("allow_delegation"))

    def test_hierarchical_enables_manager_and_delegation(self):
        _FakeCrew.last_kwargs = None
        result = StubCrewAIAgent(hierarchical=True).run_research_pipeline("tema")

        self.assertEqual(_FakeCrew.last_kwargs["process"], _FakeProcess.hierarchical)
        self.assertIn("manager_llm", _FakeCrew.last_kwargs)
        self.assertIsInstance(_FakeCrew.last_kwargs["manager_llm"], _FakeLLM)
        self.assertTrue(result["delegation"]["enabled"])
        self.assertEqual(result["delegation"]["process"], "hierarchical")
        # Agentes com delegação habilitada no modo hierárquico.
        for agent in _FakeCrew.last_kwargs["agents"]:
            self.assertTrue(agent.kwargs.get("allow_delegation"))


if __name__ == "__main__":
    unittest.main()
