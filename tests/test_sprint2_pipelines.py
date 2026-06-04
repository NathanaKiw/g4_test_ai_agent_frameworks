"""Unit tests for LangGraph and CrewAI pipeline behaviors.

These tests validate the LangGraph orchestration contract (three-stage
pipeline with timings) and CrewAI-specific metrics and error handling
for insufficient quota scenarios.
"""

import unittest

from test_crewai.research_agent import (
    CrewAIResearchReportAgent,
    _is_insufficient_quota_error,
)
from test_langgraph.research_agent import LangGraphResearchReportAgent


class StubResearchService:
    def insert_research_report(self, report_data):
        return None

    def close(self):
        return None


class StubLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class StubLangGraphAgent(LangGraphResearchReportAgent):
    """A LangGraph agent variant that stubs LLM calls for deterministic tests."""

    def __init__(self):
        self._last_api_calls = 0
        self.research_service = StubResearchService()
        self.logger = StubLogger()
        self.graph = self._build_graph()

    def _chat(self, system_content, user_content):
        # Return predictable stub responses and count API call attempts.
        self._last_api_calls += 1
        return f"stub response {self._last_api_calls}"


class LangGraphPipelineTest(unittest.TestCase):
    def test_pipeline_returns_standard_benchmark_result(self):
        """LangGraph pipeline should produce the same benchmark-shaped result
        as the vanilla baseline: three LLM calls, framework metadata and
        per-stage timings.
        """
        result = StubLangGraphAgent().run_research_pipeline("tema teste")

        self.assertEqual(result["framework"], "LangGraph")
        self.assertEqual(result["api_calls"], 3)
        self.assertEqual(
            set(result["stage_timings"]),
            {"research_s", "analysis_s", "report_s", "total_s"},
        )
        self.assertGreaterEqual(
            set(result),
            {
                "topic",
                "research",
                "analysis",
                "report",
                "timestamp",
                "framework",
                "api_calls",
                "stage_timings",
            },
        )


class CrewAIMetricsTest(unittest.TestCase):
    def test_task_callback_records_stage_timings(self):
        """Verify that the CrewAI task callback increments API call count and
        records per-stage timings when invoked for each task.
        """
        agent = object.__new__(CrewAIResearchReportAgent)
        agent._last_api_calls = 0

        stage_timings = {}
        callback = agent._task_callback_factory(stage_timings)
        callback(None)
        callback(None)
        callback(None)

        self.assertEqual(agent._last_api_calls, 3)
        self.assertEqual(
            set(stage_timings),
            {"research_s", "analysis_s", "report_s"},
        )

    def test_insufficient_quota_error_is_detected(self):
        """Detect when a raised exception contains an `insufficient_quota`
        error code in its `body` payload.
        """
        class FakeRateLimitError(Exception):
            pass

        exc = FakeRateLimitError()
        exc.body = {"error": {"code": "insufficient_quota"}}

        self.assertTrue(_is_insufficient_quota_error(exc))

    def test_run_research_pipeline_wraps_insufficient_quota(self):
        """If the Crew kickoff raises an insufficient-quota error, the
        pipeline should raise a `ValueError` with a clear message.
        """
        class FakeRateLimitError(Exception):
            pass

        class FakeCrew:
            def __init__(self, *args, **kwargs):
                return None

            def kickoff(self, inputs):
                exc = FakeRateLimitError()
                exc.body = {"error": {"code": "insufficient_quota"}}
                raise exc

        class StubCrewAIAgent(CrewAIResearchReportAgent):
            def __init__(self):
                self._last_api_calls = 0
                self.research_service = StubResearchService()
                self.logger = StubLogger()
                self.Crew = FakeCrew
                self.Process = type("Process", (), {"sequential": object()})

            def _build_tasks(self, topic, stage_timings):
                task = type("Task", (), {"output": type("Output", (), {"raw": "stub"})()})()
                return [task, task, task], [object(), object(), object()]

        with self.assertRaisesRegex(ValueError, "Quota da API Groq esgotada"):
            StubCrewAIAgent().run_research_pipeline("tema teste")


if __name__ == "__main__":
    unittest.main()
