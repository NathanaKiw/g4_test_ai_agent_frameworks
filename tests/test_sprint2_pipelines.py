import unittest

from test_crewai.research_agent import CrewAIResearchReportAgent
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
    def __init__(self):
        self._last_api_calls = 0
        self.research_service = StubResearchService()
        self.logger = StubLogger()
        self.graph = self._build_graph()

    def _chat(self, system_content, user_content):
        self._last_api_calls += 1
        return f"stub response {self._last_api_calls}"


class LangGraphPipelineTest(unittest.TestCase):
    def test_pipeline_returns_standard_benchmark_result(self):
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


if __name__ == "__main__":
    unittest.main()
