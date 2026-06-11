"""Stubs mínimos para `langchain_core.messages` usados nos pipelines."""
from dataclasses import dataclass


@dataclass
class HumanMessage:
    content: str


@dataclass
class SystemMessage:
    content: str
