"""Baseline vanilla — chamadas diretas à API OpenAI."""

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

from .research_agent import ResearchReportAgent

__all__ = ["ResearchReportAgent"]
