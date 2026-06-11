"""Engenharia de contexto: compactação de histórico e notas estruturadas.

Estratégias usadas para reduzir o contexto repassado entre as etapas do
pipeline, mantendo apenas a informação de maior valor:

- **Compactação de histórico** (`compact_history`): quando o texto de uma
  etapa anterior excede um orçamento de caracteres, ele é condensado mantendo
  títulos, listas e frases de alto sinal (números, anos, palavras-chave),
  reduzindo tokens repassados adiante.
- **Notas estruturadas** (`extract_structured_notes` / `render_notes`):
  um bloco de anotações acumulado etapa a etapa, funcionando como uma memória
  de trabalho compacta entregue às etapas seguintes em lugar do texto bruto.

Tudo é determinístico (sem chamadas extras ao LLM), preservando o contrato de
`api_calls` do benchmark e tornando os resultados reproduzíveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List


# Palavras que sinalizam frases de alto valor informativo.
_SIGNAL_KEYWORDS = (
    "porque",
    "portanto",
    "risco",
    "tend[eê]ncia",
    "oportunidade",
    "desafio",
    "impacto",
    "recomenda",
    "conclus",
    "principal",
    "crescimento",
    "redu[cç][aã]o",
)
_SIGNAL_RE = re.compile("|".join(_SIGNAL_KEYWORDS), re.IGNORECASE)
_HAS_NUMBER_RE = re.compile(r"\d")
# Linhas de título markdown ou itens de lista.
_HEADER_OR_BULLET_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)")


@dataclass
class CompactionResult:
    """Resultado da compactação de um texto.

    Attributes:
        text: Texto compactado (ou original, se já dentro do orçamento).
        original_chars: Tamanho original em caracteres.
        compacted_chars: Tamanho após a compactação.
        compacted: Indica se houve compactação efetiva.
    """

    text: str
    original_chars: int
    compacted_chars: int
    compacted: bool

    @property
    def ratio(self) -> float:
        """Fração do conteúdo preservada (1.0 = nada removido)."""
        if not self.original_chars:
            return 1.0
        return round(self.compacted_chars / self.original_chars, 3)

    def to_dict(self) -> dict:
        return {
            "original_chars": self.original_chars,
            "compacted_chars": self.compacted_chars,
            "ratio": self.ratio,
            "compacted": self.compacted,
        }


@dataclass
class StructuredNotes:
    """Memória de trabalho acumulada ao longo do pipeline."""

    entries: List[dict] = field(default_factory=list)

    def add(self, stage: str, points: List[str]) -> None:
        if points:
            self.entries.append({"stage": stage, "points": points})

    @property
    def total_points(self) -> int:
        return sum(len(entry["points"]) for entry in self.entries)

    def to_dict(self) -> dict:
        return {"entries": list(self.entries), "total_points": self.total_points}


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def extract_structured_notes(stage: str, text: str, *, max_points: int = 6) -> List[str]:
    """Extrai os pontos-chave de uma etapa para a memória estruturada.

    Prioriza títulos e itens de lista; em seguida, frases que contenham números
    ou palavras de alto sinal. Mantém a ordem de aparição e remove duplicatas.

    Args:
        stage: Nome da etapa de origem (ex.: ``"pesquisa"``).
        text: Texto produzido pela etapa.
        max_points: Número máximo de pontos retornados.

    Returns:
        Lista de pontos-chave normalizados (sem marcadores de lista).
    """
    if not text:
        return []

    points: List[str] = []
    seen: set[str] = set()

    def _push(candidate: str) -> None:
        cleaned = _HEADER_OR_BULLET_RE.sub("", candidate).strip(" #-*•\t")
        cleaned = re.sub(r"\s+", " ", cleaned)
        key = cleaned.lower()
        if cleaned and key not in seen and len(cleaned) > 3:
            seen.add(key)
            points.append(cleaned)

    # 1ª passada: títulos e itens de lista (estrutura explícita do texto).
    for line in text.splitlines():
        if len(points) >= max_points:
            break
        if _HEADER_OR_BULLET_RE.match(line):
            _push(line)

    # 2ª passada: frases de alto sinal, se ainda houver espaço.
    if len(points) < max_points:
        for sentence in _split_sentences(text):
            if len(points) >= max_points:
                break
            if _HAS_NUMBER_RE.search(sentence) or _SIGNAL_RE.search(sentence):
                _push(sentence)

    return points[:max_points]


def render_notes(notes: StructuredNotes) -> str:
    """Renderiza as notas estruturadas como um bloco compacto de markdown."""
    if not notes.entries:
        return ""

    lines = ["## Notas estruturadas (memória do pipeline)"]
    for entry in notes.entries:
        lines.append(f"### {entry['stage'].capitalize()}")
        lines.extend(f"- {point}" for point in entry["points"])
    return "\n".join(lines)


def compact_history(text: str, *, max_chars: int = 1500) -> CompactionResult:
    """Compacta um texto longo respeitando um orçamento de caracteres.

    Se o texto já cabe no orçamento, é retornado intacto. Caso contrário,
    preserva primeiro títulos/listas e frases de alto sinal, na ordem original,
    até atingir o limite, e acrescenta um marcador de truncamento.

    Args:
        text: Texto a compactar (tipicamente a saída de uma etapa anterior).
        max_chars: Orçamento máximo de caracteres do resultado.

    Returns:
        ``CompactionResult`` com o texto e as métricas de compactação.
    """
    source = text or ""
    original_chars = len(source)

    if original_chars <= max_chars:
        return CompactionResult(
            text=source,
            original_chars=original_chars,
            compacted_chars=original_chars,
            compacted=False,
        )

    high_signal: List[str] = []
    others: List[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADER_OR_BULLET_RE.match(line) or _HAS_NUMBER_RE.search(stripped) or _SIGNAL_RE.search(stripped):
            high_signal.append(stripped)
        else:
            others.append(stripped)

    marker = "\n\n[...histórico compactado por engenharia de contexto...]"
    budget = max(0, max_chars - len(marker))

    selected: List[str] = []
    used = 0
    for line in high_signal + others:
        addition = len(line) + 1  # +1 pela quebra de linha
        if used + addition > budget:
            continue
        selected.append(line)
        used += addition

    compacted_text = "\n".join(selected) + marker
    return CompactionResult(
        text=compacted_text,
        original_chars=original_chars,
        compacted_chars=len(compacted_text),
        compacted=True,
    )
