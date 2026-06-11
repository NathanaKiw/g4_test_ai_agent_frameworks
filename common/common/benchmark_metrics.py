"""Utilities for benchmark token usage and transparency assessment."""

from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import mean
from typing import Any, Iterable, Sequence


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _count_estimated_tokens(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    return max(1, round(len(cleaned) / 4))


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    source: str = "missing"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "source": self.source,
        }


def _build_usage(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cached_prompt_tokens: int = 0,
    source: str = "actual",
) -> TokenUsage:
    if not total_tokens and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        source=source,
    )


def _extract_usage_from_mapping(mapping: Any) -> TokenUsage:
    if not isinstance(mapping, dict):
        return TokenUsage()

    nested = mapping.get("token_usage") or mapping.get("usage")
    if isinstance(nested, dict):
        candidate = _extract_usage_from_mapping(nested)
        if candidate.total_tokens or candidate.prompt_tokens or candidate.completion_tokens:
            return candidate

    prompt_tokens = _coerce_int(mapping.get("prompt_tokens", mapping.get("input_tokens")))
    completion_tokens = _coerce_int(mapping.get("completion_tokens", mapping.get("output_tokens")))
    total_tokens = _coerce_int(mapping.get("total_tokens"))
    cached_prompt_tokens = _coerce_int(mapping.get("cached_prompt_tokens"))

    if not any((prompt_tokens, completion_tokens, total_tokens, cached_prompt_tokens)):
        return TokenUsage()

    return _build_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
    )


def _extract_usage_from_object(source: Any) -> TokenUsage:
    if source is None:
        return TokenUsage()

    for attribute in ("usage_metadata", "response_metadata", "usage", "token_usage"):
        if hasattr(source, attribute):
            candidate = extract_token_usage(getattr(source, attribute))
            if candidate.total_tokens or candidate.prompt_tokens or candidate.completion_tokens:
                return candidate

    prompt_tokens = _coerce_int(getattr(source, "prompt_tokens", 0))
    completion_tokens = _coerce_int(getattr(source, "completion_tokens", 0))
    total_tokens = _coerce_int(getattr(source, "total_tokens", 0))
    cached_prompt_tokens = _coerce_int(getattr(source, "cached_prompt_tokens", 0))

    if not any((prompt_tokens, completion_tokens, total_tokens, cached_prompt_tokens)):
        return TokenUsage()

    return _build_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
    )


def extract_token_usage(source: Any) -> TokenUsage:
    if source is None:
        return TokenUsage()

    if isinstance(source, TokenUsage):
        return source

    if isinstance(source, dict):
        candidate = _extract_usage_from_mapping(source)
        return candidate if candidate.total_tokens or candidate.prompt_tokens or candidate.completion_tokens else TokenUsage()

    candidate = _extract_usage_from_object(source)
    return candidate if candidate.total_tokens or candidate.prompt_tokens or candidate.completion_tokens else TokenUsage()


def estimate_token_usage(prompt_text: str = "", completion_text: str = "") -> TokenUsage:
    prompt_tokens = _count_estimated_tokens(prompt_text)
    completion_tokens = _count_estimated_tokens(completion_text)
    return _build_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        source="estimated",
    )


def aggregate_token_usages(usages: Iterable[TokenUsage]) -> TokenUsage:
    items = list(usages)
    if not items:
        return TokenUsage()

    source = "actual"
    if any(item.source != "actual" for item in items):
        source = "mixed" if any(item.source == "actual" for item in items) else "estimated"

    return TokenUsage(
        prompt_tokens=sum(item.prompt_tokens for item in items),
        completion_tokens=sum(item.completion_tokens for item in items),
        total_tokens=sum(item.total_tokens for item in items),
        cached_prompt_tokens=sum(item.cached_prompt_tokens for item in items),
        source=source,
    )


def _keyword_hits(text: str, keywords: Sequence[str]) -> list[str]:
    normalized = _normalize_text(text)
    return [keyword for keyword in keywords if keyword in normalized]


def _structure_signals(text: str) -> list[str]:
    signals: list[str] = []
    if re.search(r"(^|\n)\s*(?:#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)", text):
        signals.append("estrutura explícita")
    if "```" in text:
        signals.append("blocos delimitados")
    return signals


def assess_transparency(stage_name: str, system_prompt: str, user_prompt: str, output_text: str) -> dict[str, Any]:
    combined_text = f"{system_prompt}\n{user_prompt}\n{output_text}"
    signals = _structure_signals(output_text)
    signals.extend(_keyword_hits(combined_text, [
        "plano",
        "passos",
        "metodologia",
        "abordagem",
        "critério",
        "critérios",
        "premissa",
        "assunção",
        "hipótese",
        "riscos",
        "lacunas",
        "limitação",
        "limitações",
        "incerteza",
        "próximos passos",
        "recomendação",
        "recomendações",
        "conclusão",
    ]))

    score = 0
    if any(signal == "estrutura explícita" for signal in signals):
        score += 1
    if _keyword_hits(combined_text, ["metodologia", "abordagem", "critério", "critérios", "passos", "plano"]):
        score += 1
    if _keyword_hits(combined_text, ["assunção", "premissa", "hipótese", "incerteza"]):
        score += 1
    if _keyword_hits(combined_text, ["limitação", "limitações", "lacuna", "lacunas", "riscos"]):
        score += 1
    if _keyword_hits(combined_text, ["próximos passos", "recomendação", "recomendações", "ação", "ações"]):
        score += 1

    label = "baixa"
    if score >= 4:
        label = "alta"
    elif score >= 2:
        label = "moderada"

    missing = []
    if not _keyword_hits(combined_text, ["limitação", "limitações", "lacuna", "lacunas"]):
        missing.append("limitações/lacunas")
    if not _keyword_hits(combined_text, ["plano", "passos", "metodologia", "abordagem"]):
        missing.append("plano ou método explícito")
    if not _keyword_hits(combined_text, ["próximos passos", "recomendação", "recomendações"]):
        missing.append("próximos passos")

    summary_parts = [
        f"{stage_name}: transparência {label} ({score}/5).",
    ]
    if signals:
        summary_parts.append("Sinais observados: " + ", ".join(dict.fromkeys(signals)) + ".")
    if missing:
        summary_parts.append("Ausências relevantes: " + ", ".join(missing) + ".")
    else:
        summary_parts.append("O texto explicita estrutura e pontos de decisão observáveis.")

    return {
        "stage": stage_name,
        "score": score,
        "label": label,
        "summary": " ".join(summary_parts),
        "signals": list(dict.fromkeys(signals)),
        "missing": missing,
    }


def summarize_transparency(assessments: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(assessments)
    if not items:
        return {
            "score": 0.0,
            "label": "baixa",
            "summary": "Sem avaliações de transparência disponíveis.",
            "stages": [],
        }

    avg_score = mean(item.get("score", 0) for item in items)
    label = "baixa"
    if avg_score >= 4:
        label = "alta"
    elif avg_score >= 2:
        label = "moderada"

    stage_labels = ", ".join(f"{item['stage']}: {item['label']}" for item in items)
    summary = (
        f"Transparência média {label} ({avg_score:.1f}/5). "
        f"Distribuição por etapa: {stage_labels}."
    )
    return {
        "score": round(avg_score, 2),
        "label": label,
        "summary": summary,
        "stages": items,
    }