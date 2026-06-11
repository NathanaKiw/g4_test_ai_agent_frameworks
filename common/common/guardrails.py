"""Guardrails (filtros de segurança) reutilizáveis entre frameworks.

Os guardrails atuam de forma transversal ao pipeline em dois pontos:

- **Entrada**: valida o tópico antes de gastar chamadas de API
  (vazio, tamanho excessivo, tentativas de prompt injection e conteúdo
  manifestamente proibido).
- **Saída**: higieniza o texto gerado pelo modelo, redigindo segredos que
  porventura vazem (chaves de API, tokens) antes de persistir ou exibir.

São implementados com heurísticas determinísticas (sem chamadas extras ao
LLM), preservando o contrato de `api_calls` do benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Pattern, Tuple


class GuardrailError(ValueError):
    """Erro levantado quando a entrada viola um guardrail bloqueante."""


@dataclass
class GuardrailResult:
    """Resultado da aplicação de um guardrail.

    Attributes:
        allowed: Indica se o conteúdo pode seguir no pipeline.
        text: Texto possivelmente higienizado (entrada ou saída).
        violations: Lista de rótulos das violações detectadas.
        redactions: Número de trechos redigidos na saída.
    """

    allowed: bool
    text: str
    violations: List[str] = field(default_factory=list)
    redactions: int = 0

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "violations": list(self.violations),
            "redactions": self.redactions,
        }


# Tentativas comuns de manipular o agente (prompt injection / jailbreak),
# em português e inglês.
_INJECTION_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"ignore (as |todas as |suas )?(instru[cç][oõ]es|regras)", re.IGNORECASE),
    re.compile(r"ignore (the |all |your )?(previous |above )?(instructions|rules)", re.IGNORECASE),
    re.compile(r"desconsidere (as |o |suas )?(instru[cç][oõ]es|prompt|regras)", re.IGNORECASE),
    re.compile(r"(reveal|mostre|exiba|imprima|repita).{0,30}(system prompt|prompt do sistema|suas instru[cç][oõ]es)", re.IGNORECASE),
    re.compile(r"\bjailbreak\b|\bDAN\b|developer mode|modo desenvolvedor", re.IGNORECASE),
    re.compile(r"act as if|aja como se|finja que voc[eê]", re.IGNORECASE),
)

# Conteúdo manifestamente proibido — lista ilustrativa e conservadora.
_BLOCKED_CONTENT_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"como (fabricar|construir|fazer).{0,40}(bomba|explosivo|arma de fogo)", re.IGNORECASE),
    re.compile(r"(fabricar|sintetizar|produzir).{0,30}(metanfetamina|drogas il[ií]citas|ant?raz)", re.IGNORECASE),
    re.compile(r"\b(child sexual abuse|csam|pornografia infantil)\b", re.IGNORECASE),
)

# Segredos que nunca devem aparecer na saída.
_SECRET_PATTERNS: Tuple[Tuple[Pattern[str], str], ...] = (
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"), "[REDACTED_GROQ_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\b(?:ghp|gho|ghs)_[A-Za-z0-9]{30,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b"), "Bearer [REDACTED_TOKEN]"),
)


def check_input(topic: str, *, max_chars: int = 2000) -> GuardrailResult:
    """Valida o tópico de pesquisa antes de iniciar o pipeline.

    Args:
        topic: Tópico informado pelo usuário.
        max_chars: Tamanho máximo aceitável para o tópico.

    Returns:
        ``GuardrailResult`` com ``allowed=False`` e a lista de violações quando
        a entrada é rejeitada.
    """
    text = (topic or "").strip()
    violations: List[str] = []

    if not text:
        violations.append("entrada_vazia")
        return GuardrailResult(allowed=False, text=text, violations=violations)

    if len(text) > max_chars:
        violations.append("entrada_muito_longa")

    if any(pattern.search(text) for pattern in _INJECTION_PATTERNS):
        violations.append("prompt_injection")

    if any(pattern.search(text) for pattern in _BLOCKED_CONTENT_PATTERNS):
        violations.append("conteudo_proibido")

    return GuardrailResult(allowed=not violations, text=text, violations=violations)


def sanitize_output(text: str) -> GuardrailResult:
    """Higieniza a saída do modelo, redigindo segredos detectados.

    A saída nunca é bloqueada — apenas higienizada — para não descartar um
    relatório útil. As redações são contabilizadas em ``redactions``.
    """
    sanitized = text or ""
    redactions = 0
    violations: List[str] = []

    for pattern, replacement in _SECRET_PATTERNS:
        sanitized, count = pattern.subn(replacement, sanitized)
        redactions += count

    if redactions:
        violations.append("segredo_redigido")

    return GuardrailResult(
        allowed=True,
        text=sanitized,
        violations=violations,
        redactions=redactions,
    )
