import unittest
from unittest.mock import patch

from common.mongodb.research_data import ResearchDataService
from test_vanilla.research_agent import (
    ResearchReportAgent,
    _is_retryable_openai_error,
)


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


class StubResearchReportAgent(ResearchReportAgent):
    def __init__(self):
        self._last_api_calls = 0
        self.research_service = StubResearchService()
        self.logger = StubLogger()

    def _chat(self, system_content, user_content):
        self._last_api_calls += 1
        return f"stub response {self._last_api_calls}"


class VanillaPipelineTest(unittest.TestCase):
    def test_pipeline_returns_standard_benchmark_result(self):
        result = StubResearchReportAgent().run_research_pipeline("tema teste")

        self.assertEqual(result["framework"], "Vanilla (OpenAI API)")
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


class ResearchDataServiceTest(unittest.TestCase):
    def test_mongodb_is_disabled_when_uri_is_not_configured(self):
        with patch.dict("os.environ", {"MONGODB_URI": ""}):
            service = ResearchDataService()

        self.assertFalse(service._enabled)
        self.assertIsNone(service.client)
        self.assertIsNone(service.collection)


class RetryPolicyTest(unittest.TestCase):
    def test_insufficient_quota_rate_limit_is_not_retryable(self):
        class FakeRateLimitError(Exception):
            pass

        exc = FakeRateLimitError()
        exc.body = {"error": {"code": "insufficient_quota"}}

        with patch("test_vanilla.research_agent.RateLimitError", FakeRateLimitError):
            self.assertFalse(_is_retryable_openai_error(exc))


if __name__ == "__main__":
    unittest.main()
