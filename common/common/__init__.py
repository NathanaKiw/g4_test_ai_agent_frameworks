"""Biblioteca compartilhada — caso de uso Agente de Pesquisa e Relatório."""

from .config import BaseConfig
from .logging_config import LoggingConfig, get_logger, setup_logging
from .mongodb.research_data import ResearchDataService
from .research_prompts import (
    analyst_messages,
    analyst_system,
    analyst_user,
    build_result,
    report_writer_messages,
    report_writer_system,
    report_writer_user,
    researcher_messages,
    researcher_system,
    researcher_user,
)

__version__ = "0.2.0"
__all__ = [
    "BaseConfig",
    "ResearchDataService",
    "LoggingConfig",
    "setup_logging",
    "get_logger",
    "researcher_system",
    "researcher_user",
    "researcher_messages",
    "analyst_system",
    "analyst_user",
    "analyst_messages",
    "report_writer_system",
    "report_writer_user",
    "report_writer_messages",
    "build_result",
]
