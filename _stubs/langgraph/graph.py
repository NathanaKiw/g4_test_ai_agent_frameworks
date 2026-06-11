"""Stub simplificado do `langgraph.graph` para testes locais.

Suporta apenas grafos lineares como usados no projeto: START -> A -> B -> C -> END.
"""
from typing import Callable, Dict, Optional

START = "__START__"
END = "__END__"


class StateGraph:
    def __init__(self, state_type=None):
        self._nodes: Dict[str, Callable] = {}
        self._edges: Dict[str, str] = {}

    def add_node(self, name: str, fn: Callable):
        self._nodes[name] = fn

    def add_edge(self, src: str, dst: str):
        self._edges[src] = dst

    def compile(self):
        nodes = dict(self._nodes)
        edges = dict(self._edges)

        class Compiled:
            def __init__(self, nodes, edges):
                self._nodes = nodes
                self._edges = edges

            def invoke(self, state: Dict):
                cur = self._edges.get(START)
                s = dict(state)
                while cur is not None and cur != END:
                    node_fn = self._nodes.get(cur)
                    if node_fn is None:
                        break
                    out = node_fn(s)
                    if isinstance(out, dict):
                        s.update(out)
                    cur = self._edges.get(cur)
                return s

        return Compiled(nodes, edges)
