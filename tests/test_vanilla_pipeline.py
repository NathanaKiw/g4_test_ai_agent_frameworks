"""Unit tests for the vanilla research pipeline.

These tests verify the baseline pipeline contract (three LLM steps,
timings, result shape) and error detection logic for retryable
vs non-retryable API errors.
"""

import unittest
from unittest.mock import patch

from common.common import TokenUsage
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
        text = (
            f"stub response {self._last_api_calls}\n"
            "## Plano\n- passos\n- metodologia\n"
            "Limitações: lacunas conhecidas.\nPróximos passos: recomendações."
        )
        usage = TokenUsage(
            prompt_tokens=7,
            completion_tokens=11,
            total_tokens=18,
            source="actual",
        )
        return text, usage


class VanillaPipelineTest(unittest.TestCase):
    def test_pipeline_returns_standard_benchmark_result(self):
        """The pipeline should execute three LLM stages and return a
        standardized result dictionary with timings and metadata.
        """
        result = StubResearchReportAgent().run_research_pipeline("tema teste")

        self.assertEqual(result["framework"], "Vanilla (Groq API)")
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
                "token_usage",
                "stage_token_usage",
            },
        )
        self.assertEqual(result["token_usage"]["total_tokens"], 54)
        self.assertEqual(
            set(result["stage_token_usage"]),
            {"research", "analysis", "report"},
        )


class ResearchDataServiceTest(unittest.TestCase):
    def test_mongodb_is_disabled_when_uri_is_not_configured(self):
        """If `MONGODB_URI` is empty, the ResearchDataService should be
        disabled (no client/collection) to avoid accidental DB connections
        during tests or when persistence is not configured.
        """
        with patch.dict("os.environ", {"MONGODB_URI": ""}):
            service = ResearchDataService()

        self.assertFalse(service._enabled)
        self.assertIsNone(service.client)
        self.assertIsNone(service.collection)


class RetryPolicyTest(unittest.TestCase):
    def test_insufficient_quota_rate_limit_is_not_retryable(self):
        """Simulate a rate-limit exception carrying an
        `insufficient_quota` error code. The retry predicate must classify
        this as non-retryable so callers can surface a clear quota error.
        """
        class FakeRateLimitError(Exception):
            pass

        exc = FakeRateLimitError()
        exc.body = {"error": {"code": "insufficient_quota"}}

        with patch("test_vanilla.research_agent.RateLimitError", FakeRateLimitError):
            self.assertFalse(_is_retryable_openai_error(exc))


if __name__ == "__main__":
    unittest.main()
