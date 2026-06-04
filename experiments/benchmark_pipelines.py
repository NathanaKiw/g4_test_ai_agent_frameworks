"""Benchmark/demo dos três pipelines do projeto.

O experimento roda os pipelines com chamadas LLM simuladas para gerar
métricas comparáveis sem depender de API externa. Ele produz:
- CSV com métricas por execução
- resumo em Markdown
- dashboard HTML com visual mais polido
- gráficos PNG com comparativos de tempo total e por etapa

Uso:
    python experiments/benchmark_pipelines.py --runs 5 --delay 0.02
    python experiments/benchmark_pipelines.py --runs 10 --delay 0.02
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts" / "benchmark"


def _ensure_import_paths() -> None:
    import sys

    paths = [
        str(ROOT / "common"),
        str(ROOT / "vanilla"),
        str(ROOT / "langgraph_pipeline"),
        str(ROOT / "crewai_pipeline"),
    ]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)


_ensure_import_paths()

from test_vanilla.research_agent import ResearchReportAgent as VanillaAgent  # noqa: E402
from test_langgraph.research_agent import LangGraphResearchReportAgent  # noqa: E402
from test_crewai.research_agent import CrewAIResearchReportAgent  # noqa: E402


@dataclass
class RunResult:
    framework: str
    run_index: int
    total_s: float
    api_calls: int
    research_s: float
    analysis_s: float
    report_s: float


class BenchmarkResearchService:
    def insert_research_report(self, report_data: Dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class BenchmarkLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None


class BaseBenchmarkAgent:
    framework_name = "Base"

    def __init__(self, delay_s: float, jitter_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.jitter_s = jitter_s

    def _sleep(self) -> None:
        delay = self.delay_s
        if self.jitter_s:
            delay += random.uniform(-self.jitter_s, self.jitter_s)
            delay = max(0.0, delay)
        if delay:
            time.sleep(delay)


class VanillaBenchmarkAgent(BaseBenchmarkAgent, VanillaAgent):
    def __init__(self, delay_s: float, jitter_s: float = 0.0) -> None:
        BaseBenchmarkAgent.__init__(self, delay_s, jitter_s)
        self._last_api_calls = 0
        self.research_service = BenchmarkResearchService()
        self.logger = BenchmarkLogger()

    def _chat(self, system_content: str, user_content: str) -> str:
        self._last_api_calls += 1
        self._sleep()
        return f"{self.framework_name} response {self._last_api_calls}"


class LangGraphBenchmarkAgent(BaseBenchmarkAgent, LangGraphResearchReportAgent):
    def __init__(self, delay_s: float, jitter_s: float = 0.0) -> None:
        BaseBenchmarkAgent.__init__(self, delay_s, jitter_s)
        self._last_api_calls = 0
        self.research_service = BenchmarkResearchService()
        self.logger = BenchmarkLogger()
        self.graph = self._build_graph()

    def _chat(self, system_content: str, user_content: str) -> str:
        self._last_api_calls += 1
        self._sleep()
        return f"{self.framework_name} response {self._last_api_calls}"


class CrewAIBenchmarkAgent(BaseBenchmarkAgent, CrewAIResearchReportAgent):
    def __init__(self, delay_s: float, jitter_s: float = 0.0) -> None:
        BaseBenchmarkAgent.__init__(self, delay_s, jitter_s)
        self._last_api_calls = 0
        self.research_service = BenchmarkResearchService()
        self.logger = BenchmarkLogger()
        self.Crew = lambda **kwargs: FakeCrew(  # noqa: E731
            delay_s=self.delay_s,
            jitter_s=self.jitter_s,
            **kwargs,
        )
        self.Process = type("Process", (), {"sequential": object()})

    def _build_agents(self):
        class FakeAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        return FakeAgent(role="Pesquisador"), FakeAgent(role="Analista"), FakeAgent(role="Redator Executivo")

    def _build_tasks(self, topic: str, stage_timings: Dict[str, float]):
        agent1, agent2, agent3 = self._build_agents()
        callback = self._task_callback_factory(stage_timings)

        class FakeTask:
            def __init__(self, output: str, callback=None):
                self.output = type("Output", (), {"raw": output})()
                self.callback = callback

        return (
            [
                FakeTask("research", callback),
                FakeTask("analysis", callback),
                FakeTask("report", callback),
            ],
            [agent1, agent2, agent3],
        )


class FakeCrew:
    def __init__(self, agents, tasks, process, verbose=False, delay_s=0.0, jitter_s=0.0):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = verbose
        self.delay_s = delay_s
        self.jitter_s = jitter_s

    def _sleep(self) -> None:
        delay = self.delay_s
        if self.jitter_s:
            delay += random.uniform(-self.jitter_s, self.jitter_s)
            delay = max(0.0, delay)
        if delay:
            time.sleep(delay)

    def kickoff(self, inputs: Dict[str, Any]):
        # Simula execução sequencial dos três tasks e aciona callbacks.
        for task in self.tasks:
            self._sleep()
            callback = getattr(task, "callback", None)
            if callback:
                callback(task)
        return None


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def _stdev(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _summarize(results: List[RunResult]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[RunResult]] = {}
    for item in results:
        grouped.setdefault(item.framework, []).append(item)

    summary: List[Dict[str, Any]] = []
    for framework, items in grouped.items():
        summary.append(
            {
                "framework": framework,
                "runs": len(items),
                "avg_total_s": _mean(item.total_s for item in items),
                "stdev_total_s": _stdev(item.total_s for item in items),
                "avg_api_calls": _mean(item.api_calls for item in items),
                "avg_research_s": _mean(item.research_s for item in items),
                "avg_analysis_s": _mean(item.analysis_s for item in items),
                "avg_report_s": _mean(item.report_s for item in items),
            }
        )
    summary.sort(key=lambda row: row["avg_total_s"])
    return summary


def _write_csv(path: Path, results: List[RunResult], summary: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "framework", "run_index", "total_s", "api_calls", "research_s", "analysis_s", "report_s"])
        for row in results:
            writer.writerow([
                "run",
                row.framework,
                row.run_index,
                _format_float(row.total_s),
                row.api_calls,
                _format_float(row.research_s),
                _format_float(row.analysis_s),
                _format_float(row.report_s),
            ])
        for row in summary:
            writer.writerow([
                "summary",
                row["framework"],
                row["runs"],
                _format_float(row["avg_total_s"]),
                _format_float(row["avg_api_calls"]),
                _format_float(row["avg_research_s"]),
                _format_float(row["avg_analysis_s"]),
                _format_float(row["avg_report_s"]),
            ])


def _write_markdown(path: Path, summary: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Benchmark dos pipelines",
        "",
        f"Data da execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Execuções por framework: {args.runs}",
        f"Delay simulado por chamada: {args.delay:.3f}s",
        f"Jitter opcional: {args.jitter:.3f}s",
        "",
        "## Resumo",
        "",
        "| Framework | Execuções | Tempo médio total (s) | Desvio padrão (s) | API calls média |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['framework']} | {row['runs']} | {row['avg_total_s']:.4f} | {row['stdev_total_s']:.4f} | {row['avg_api_calls']:.1f} |"
        )
    lines.extend([
        "",
        "## Interpretação",
        "",
        "- `api_calls` fica em 3 porque cada pipeline faz três etapas.",
        "- O tempo total reflete o delay simulado mais o overhead de cada framework.",
        "- Os gráficos mostram as médias consolidadas por framework.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")

def _write_table_markdown(
    path: Path,
    summary: List[Dict[str, Any]],
    args: argparse.Namespace,
    results: List[RunResult],
) -> None:
    lines = [
        "# Tabela do experimento",
        "",
        f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Execuções por framework: {args.runs}",
        "",
        "| Framework | Execuções | Tempo médio total (s) | Desvio padrão (s) | API calls média | Pesquisa | Análise | Relatório |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['framework']} | {row['runs']} | {row['avg_total_s']:.4f} | {row['stdev_total_s']:.4f} | {row['avg_api_calls']:.1f} | {row['avg_research_s']:.4f} | {row['avg_analysis_s']:.4f} | {row['avg_report_s']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Total de execuções: {len(results)}",
            f"Tempo médio geral: {_mean(item.total_s for item in results):.4f}s",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html_report(
        path: Path,
        summary: List[Dict[str, Any]],
        args: argparse.Namespace,
        results: List[RunResult],
) -> None:
        total_runs = len(results)
        avg_total = _mean(item.total_s for item in results)
        fastest = summary[0] if summary else None
        slowest = summary[-1] if summary else None
        frameworks = len(summary)
        html_rows = []
        for row in summary:
                html_rows.append(
                        "<tr>"
                        f"<td>{row['framework']}</td>"
                        f"<td>{row['runs']}</td>"
                        f"<td>{row['avg_total_s']:.4f}</td>"
                        f"<td>{row['stdev_total_s']:.4f}</td>"
                        f"<td>{row['avg_api_calls']:.1f}</td>"
                        f"<td>{row['avg_research_s']:.4f}</td>"
                        f"<td>{row['avg_analysis_s']:.4f}</td>"
                        f"<td>{row['avg_report_s']:.4f}</td>"
                        "</tr>"
                )

        fastest_html = (
                f"<strong>{fastest['framework']}</strong> com {fastest['avg_total_s']:.4f}s"
                if fastest
                else "-"
        )
        slowest_html = (
                f"<strong>{slowest['framework']}</strong> com {slowest['avg_total_s']:.4f}s"
                if slowest
                else "-"
        )

        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Benchmark dos Pipelines</title>
    <style>
        :root {{
            --bg: #0b1020;
            --panel: rgba(17, 24, 39, 0.78);
            --panel-strong: rgba(15, 23, 42, 0.95);
            --line: rgba(148, 163, 184, 0.16);
            --text: #e5eefb;
            --muted: #9fb2cc;
            --accent: #7c3aed;
            --accent-2: #38bdf8;
            --success: #34d399;
            --shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.28), transparent 28%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.22), transparent 24%),
                linear-gradient(180deg, #09101f 0%, #111827 100%);
            color: var(--text);
        }}
        .wrap {{ max-width: 1280px; margin: 0 auto; padding: 34px 22px 48px; }}
        .hero {{
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.26), rgba(56, 189, 248, 0.12));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 28px;
            padding: 30px;
            box-shadow: var(--shadow);
            overflow: hidden;
            position: relative;
        }}
        .hero::after {{
            content: "";
            position: absolute;
            inset: auto -120px -120px auto;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(255,255,255,0.12), transparent 70%);
            border-radius: 999px;
        }}
        .eyebrow {{
            display: inline-flex; align-items: center; gap: 8px;
            padding: 8px 12px; border-radius: 999px;
            background: rgba(255,255,255,0.08); color: var(--muted); font-size: 13px;
            letter-spacing: .02em;
        }}
        h1 {{ margin: 16px 0 10px; font-size: clamp(30px, 4vw, 54px); line-height: 1; }}
        .lead {{ margin: 0; max-width: 860px; color: var(--muted); font-size: 16px; line-height: 1.65; }}
        .meta {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 22px; }}
        .card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 18px;
            box-shadow: var(--shadow);
        }}
        .card .label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .08em; }}
        .card .value {{ margin-top: 8px; font-size: 26px; font-weight: 700; }}
        .card .sub {{ margin-top: 6px; color: var(--muted); font-size: 13px; line-height: 1.45; }}
        .section {{ margin-top: 22px; }}
        .section h2 {{ margin: 0 0 12px; font-size: 22px; }}
        .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
        .panel {{
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 20px;
            box-shadow: var(--shadow);
        }}
        .panel img {{ width: 100%; display: block; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); }}
        table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 18px; }}
        th, td {{ text-align: left; padding: 14px 12px; border-bottom: 1px solid var(--line); }}
        th {{ color: #d8e7fb; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; }}
        td {{ color: #edf4ff; font-size: 14px; }}
        tbody tr:nth-child(odd) {{ background: rgba(255,255,255,0.03); }}
        .note {{ color: var(--muted); font-size: 14px; line-height: 1.65; }}
        .tag {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.08); color: #f8fbff; font-size: 13px; }}
        .footer {{ margin-top: 16px; color: var(--muted); font-size: 13px; text-align: center; }}
        @media (max-width: 900px) {{
            .meta, .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="wrap">
        <section class="hero">
            <div class="eyebrow">Benchmark demo • {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
            <h1>Pipelines comparados com visual de dashboard</h1>
            <p class="lead">Experimento local e reprodutível com três frameworks, usando chamadas simuladas para gerar métricas comparáveis sem depender de API externa.</p>
            <div class="meta">
                <div class="card"><div class="label">Frameworks</div><div class="value">{frameworks}</div><div class="sub">Vanilla, LangGraph e CrewAI</div></div>
                <div class="card"><div class="label">Execuções</div><div class="value">{total_runs}</div><div class="sub">{args.runs} por framework</div></div>
                <div class="card"><div class="label">Tempo médio geral</div><div class="value">{avg_total:.4f}s</div><div class="sub">Média de todos os runs</div></div>
                <div class="card"><div class="label">Mais rápido / mais lento</div><div class="value" style="font-size: 18px; line-height: 1.35;">{fastest_html}<br/>{slowest_html}</div><div class="sub">Comparação por média total</div></div>
            </div>
        </section>

        <section class="section panel">
            <h2>Resumo quantitativo</h2>
            <table>
                <thead>
                    <tr>
                        <th>Framework</th><th>Execuções</th><th>Tempo médio total (s)</th><th>Desvio padrão</th><th>API calls</th><th>Pesquisa</th><th>Análise</th><th>Relatório</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(html_rows)}
                </tbody>
            </table>
        </section>

        <div class="grid section">
            <div class="panel">
                <h2>Tempo médio total</h2>
                <img src="avg_total_time.png" alt="Gráfico de tempo médio total por framework" />
            </div>
            <div class="panel">
                <h2>Tempo médio da etapa de pesquisa</h2>
                <img src="avg_stage_time.png" alt="Gráfico da etapa de pesquisa por framework" />
            </div>
        </div>

        <section class="section panel">
            <h2>Leitura rápida</h2>
            <p class="note">• `api_calls` fica em 3 em todos os frameworks porque cada pipeline tem três etapas.</p>
            <p class="note">• O tempo total reflete o delay simulado mais o overhead estrutural de cada framework.</p>
        </section>

        <div class="footer">Gerado por <code>experiments/benchmark_pipelines.py</code> • Artefatos em <code>artifacts/benchmark/</code></div>
    </div>
</body>
</html>"""

        path.write_text(html, encoding="utf-8")


def _load_font(size: int = 18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _hex(color: str) -> tuple[int, int, int]:
    return ImageColor.getrgb(color)


def _draw_chart_background(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    for y in range(height):
        ratio = y / max(1, height - 1)
        base = int(250 - ratio * 18)
        draw.line((0, y, width, y), fill=(base, base, base + 1))


def _draw_bar_chart(path: Path, title: str, labels: List[str], values: List[float], unit: str) -> None:
    width, height = 1200, 720
    margin_x, margin_y = 100, 140
    plot_w = width - 2 * margin_x
    plot_h = height - 2 * margin_y

    colors = ["#6d28d9", "#2563eb", "#0ea5e9", "#10b981"]
    grid = _hex("#d7e2f0")
    axis = _hex("#3a4253")
    text = _hex("#132238")
    muted = _hex("#607089")

    image = Image.new("RGB", (width, height), _hex("#f7f9fc"))
    draw = ImageDraw.Draw(image)
    _draw_chart_background(draw, width, height)
    font_title = _load_font(30)
    font_axis = _load_font(16)
    font_value = _load_font(16)
    font_sub = _load_font(14)

    draw.rounded_rectangle((48, 36, width - 48, 120), radius=24, fill=_hex("#ffffff"), outline=_hex("#e6ebf2"))
    draw.text((70, 52), title, fill=text, font=font_title)
    draw.text((70, 100), "Média consolidada por framework", fill=muted, font=font_sub)

    draw.rounded_rectangle((margin_x - 18, margin_y - 18, width - margin_x + 12, margin_y + plot_h + 16), radius=28, fill=_hex("#ffffff"), outline=_hex("#e6ebf2"))
    draw.line((margin_x, margin_y, margin_x, margin_y + plot_h), fill=axis, width=2)
    draw.line((margin_x, margin_y + plot_h, margin_x + plot_w, margin_y + plot_h), fill=axis, width=2)

    if not values:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return

    max_value = max(values) * 1.22 if max(values) > 0 else 1.0
    for tick in range(1, 6):
        y = margin_y + int(plot_h - (plot_h / 5) * tick)
        draw.line((margin_x, y, margin_x + plot_w, y), fill=grid, width=1)
        tick_value = max_value * (tick / 5)
        draw.text((24, y - 10), f"{tick_value:.2f}", fill=muted, font=font_sub)

    bar_gap = 42
    bar_width = max(120, (plot_w - bar_gap * (len(values) + 1)) / len(values))
    x = margin_x + bar_gap

    for index, (label, value) in enumerate(zip(labels, values)):
        bar_height = 0 if max_value == 0 else int((value / max_value) * plot_h)
        top = margin_y + plot_h - bar_height
        color = _hex(colors[index % len(colors)])
        draw.rounded_rectangle((x, top, x + bar_width, margin_y + plot_h), radius=18, fill=color)
        draw.rounded_rectangle((x, top, x + bar_width, margin_y + plot_h), radius=18, outline=_hex("#ffffff"), width=2)
        draw.text((x + 8, top - 28), f"{value:.4f} {unit}", fill=text, font=font_value)
        draw.text((x, margin_y + plot_h + 14), label, fill=axis, font=font_axis)
        x += bar_width + bar_gap

    draw.text((margin_x, height - 42), f"Escala máxima: {max_value:.4f} {unit}", fill=muted, font=font_sub)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _draw_grouped_stage_chart(path: Path, summary: List[Dict[str, Any]]) -> None:
    width, height = 1200, 760
    margin_x, margin_y = 90, 150
    plot_w = width - 2 * margin_x
    plot_h = height - 2 * margin_y

    categories = ["research_s", "analysis_s", "report_s"]
    labels = ["Pesquisa", "Análise", "Relatório"]
    palette = {
        "Vanilla": ["#6d28d9", "#8b5cf6", "#c4b5fd"],
        "LangGraph": ["#2563eb", "#60a5fa", "#bfdbfe"],
        "CrewAI": ["#0ea5e9", "#22c55e", "#86efac"],
    }

    image = Image.new("RGB", (width, height), _hex("#f7f9fc"))
    draw = ImageDraw.Draw(image)
    _draw_chart_background(draw, width, height)
    font_title = _load_font(30)
    font_axis = _load_font(16)
    font_small = _load_font(13)

    draw.rounded_rectangle((48, 36, width - 48, 124), radius=24, fill=_hex("#ffffff"), outline=_hex("#e6ebf2"))
    draw.text((70, 52), "Comparativo por etapa", fill=_hex("#132238"), font=font_title)
    draw.text((70, 100), "Média de tempo por etapa em cada framework", fill=_hex("#607089"), font=font_small)

    draw.rounded_rectangle((margin_x - 18, margin_y - 18, width - margin_x + 12, margin_y + plot_h + 20), radius=28, fill=_hex("#ffffff"), outline=_hex("#e6ebf2"))
    draw.line((margin_x, margin_y, margin_x, margin_y + plot_h), fill=_hex("#3a4253"), width=2)
    draw.line((margin_x, margin_y + plot_h, margin_x + plot_w, margin_y + plot_h), fill=_hex("#3a4253"), width=2)

    max_value = max((row[f"avg_{cat}"] for row in summary for cat in categories), default=1.0)
    max_value *= 1.25
    for tick in range(1, 6):
        y = margin_y + int(plot_h - (plot_h / 5) * tick)
        draw.line((margin_x, y, margin_x + plot_w, y), fill=_hex("#d7e2f0"), width=1)
        draw.text((24, y - 10), f"{max_value * (tick / 5):.2f}", fill=_hex("#607089"), font=font_small)

    framework_count = len(summary)
    category_gap = 28
    group_gap = 50
    usable_w = plot_w - group_gap * (framework_count - 1)
    group_w = usable_w / max(1, framework_count)
    bar_w = (group_w - category_gap * 2) / len(categories)
    x = margin_x

    for row in summary:
        framework = row["framework"]
        row_colors = palette.get(framework, ["#7c3aed", "#8b5cf6", "#c4b5fd"])
        for idx, cat in enumerate(categories):
            value = float(row[f"avg_{cat}"])
            bar_h = 0 if max_value == 0 else int((value / max_value) * plot_h)
            left = x + idx * (bar_w + 10)
            top = margin_y + plot_h - bar_h
            fill = _hex(row_colors[idx % len(row_colors)])
            draw.rounded_rectangle((left, top, left + bar_w, margin_y + plot_h), radius=14, fill=fill)
            draw.text((left, top - 22), f"{value:.4f}s", fill=_hex("#132238"), font=font_small)
        draw.text((x, margin_y + plot_h + 14), framework, fill=_hex("#3a4253"), font=font_axis)
        x += group_w + group_gap

    legend_x = margin_x
    legend_y = height - 46
    legend_items = [
        ("Pesquisa", "#6d28d9"),
        ("Análise", "#2563eb"),
        ("Relatório", "#0ea5e9"),
    ]
    for label, color in legend_items:
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), radius=6, fill=_hex(color))
        draw.text((legend_x + 26, legend_y - 2), label, fill=_hex("#3a4253"), font=font_axis)
        legend_x += 150

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _draw_stage_matrix_chart(path: Path, summary: List[Dict[str, Any]]) -> None:
    width, height = 1200, 620
    margin_x, margin_y = 130, 150
    plot_w = width - 2 * margin_x
    plot_h = height - 2 * margin_y

    stages = [
        ("research_s", "Pesquisa", "#6d28d9"),
        ("analysis_s", "Análise", "#2563eb"),
        ("report_s", "Relatório", "#0ea5e9"),
    ]

    image = Image.new("RGB", (width, height), _hex("#f7f9fc"))
    draw = ImageDraw.Draw(image)
    _draw_chart_background(draw, width, height)
    font_title = _load_font(30)
    font_axis = _load_font(16)
    font_small = _load_font(13)

    draw.rounded_rectangle((48, 36, width - 48, 124), radius=24, fill=_hex("#ffffff"), outline=_hex("#e6ebf2"))
    draw.text((70, 52), "Comparação lado a lado das etapas", fill=_hex("#132238"), font=font_title)
    draw.text((70, 102), "Cada grupo mostra pesquisa, análise e relatório no mesmo framework", fill=_hex("#607089"), font=font_small)

    draw.rounded_rectangle((margin_x - 18, margin_y - 18, width - margin_x + 12, margin_y + plot_h + 18), radius=28, fill=_hex("#ffffff"), outline=_hex("#e6ebf2"))
    draw.line((margin_x, margin_y, margin_x, margin_y + plot_h), fill=_hex("#3a4253"), width=2)
    draw.line((margin_x, margin_y + plot_h, margin_x + plot_w, margin_y + plot_h), fill=_hex("#3a4253"), width=2)

    max_value = max((row[f"avg_{key}"] for row in summary for key, _, _ in stages), default=1.0)
    max_value *= 1.25
    for tick in range(1, 6):
        y = margin_y + int(plot_h - (plot_h / 5) * tick)
        draw.line((margin_x, y, margin_x + plot_w, y), fill=_hex("#d7e2f0"), width=1)
        draw.text((34, y - 10), f"{max_value * (tick / 5) * 1000:.0f} ms", fill=_hex("#607089"), font=font_small)

    framework_count = len(summary)
    group_w = plot_w / max(1, framework_count)
    bar_w = min(72, (group_w - 40) / len(stages))
    x = margin_x + 18

    for row in summary:
        group_center = x + group_w / 2
        start_x = group_center - ((len(stages) * bar_w) + (len(stages) - 1) * 18) / 2
        for idx, (key, _, color) in enumerate(stages):
            value = float(row[f"avg_{key}"])
            bar_h = 0 if max_value == 0 else int((value / max_value) * plot_h)
            left = start_x + idx * (bar_w + 18)
            top = margin_y + plot_h - bar_h
            fill = _hex(color)
            draw.rounded_rectangle((left, top, left + bar_w, margin_y + plot_h), radius=12, fill=fill)
            draw.text((left - 10, top - 22), f"{value * 1000:.1f} ms", fill=_hex("#132238"), font=font_small)
        draw.text((x + 36, margin_y + plot_h + 14), row["framework"], fill=_hex("#3a4253"), font=font_axis)
        x += group_w

    legend_x = margin_x
    legend_y = height - 46
    for _, label, color in stages:
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), radius=6, fill=_hex(color))
        draw.text((legend_x + 26, legend_y - 2), label, fill=_hex("#3a4253"), font=font_axis)
        legend_x += 160

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _run_one(agent_factory: Callable[[], Any], framework: str, run_index: int, topic: str) -> RunResult:
    agent = agent_factory()
    result = agent.run_research_pipeline(topic)
    timings = result.get("stage_timings", {})
    return RunResult(
        framework=framework,
        run_index=run_index,
        total_s=float(timings.get("total_s", 0.0)),
        api_calls=int(result.get("api_calls", 0)),
        research_s=float(timings.get("research_s", 0.0)),
        analysis_s=float(timings.get("analysis_s", 0.0)),
        report_s=float(timings.get("report_s", 0.0)),
    )


def _factory_map(delay: float, jitter: float) -> Dict[str, Callable[[], Any]]:
    return {
        "Vanilla": lambda: VanillaBenchmarkAgent(delay, jitter),
        "LangGraph": lambda: LangGraphBenchmarkAgent(delay, jitter),
        "CrewAI": lambda: CrewAIBenchmarkAgent(delay, jitter),
    }


def run_benchmark(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = []
    factories = _factory_map(args.delay, args.jitter)
    topic = args.topic

    for framework, factory in factories.items():
        for run_index in range(1, args.runs + 1):
            results.append(_run_one(factory, framework, run_index, topic))

    summary = _summarize(results)

    _write_csv(ARTIFACTS_DIR / "benchmark_results.csv", results, summary)
    _write_markdown(ARTIFACTS_DIR / "benchmark_report.md", summary, args)
    _write_table_markdown(ARTIFACTS_DIR / "benchmark_table.md", summary, args, results)
    _write_html_report(ARTIFACTS_DIR / "benchmark_dashboard.html", summary, args, results)

    _draw_bar_chart(
        ARTIFACTS_DIR / "avg_total_time.png",
        "Tempo médio total por framework",
        [row["framework"] for row in summary],
        [row["avg_total_s"] for row in summary],
        "s",
    )
    _draw_grouped_stage_chart(ARTIFACTS_DIR / "avg_stage_time.png", summary)
    _draw_stage_matrix_chart(ARTIFACTS_DIR / "stage_side_by_side.png", summary)

    print(f"Benchmark concluído. Artefatos em: {ARTIFACTS_DIR}")
    for row in summary:
        print(
            f"- {row['framework']}: avg_total={row['avg_total_s']:.4f}s, "
            f"api_calls={row['avg_api_calls']:.1f}, runs={row['runs']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark/demo dos pipelines do projeto")
    parser.add_argument("--runs", type=int, default=10, help="Número de execuções por framework")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay simulado por chamada LLM em segundos")
    parser.add_argument("--jitter", type=float, default=0.005, help="Variação aleatória opcional no delay")
    parser.add_argument("--topic", type=str, default="Impacto da IA na educação brasileira", help="Tópico do benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
