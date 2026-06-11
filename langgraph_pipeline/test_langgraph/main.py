"""CLI do protótipo LangGraph."""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

from common.common import GuardrailError  # noqa: E402

from .research_agent import LangGraphResearchReportAgent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente de Pesquisa — LangGraph")
    parser.add_argument("--topic", required=True, help="Tópico de pesquisa")
    args = parser.parse_args()

    agent = None
    try:
        agent = LangGraphResearchReportAgent()
        result = agent.run_research_pipeline(args.topic)

        print("\n=== RELATÓRIO FINAL (LangGraph) ===\n")
        print(result["report"])

        timings = result.get("stage_timings", {})
        print(f"\n[Chamadas API: {result.get('api_calls', 3)}]")
        if timings:
            print(
                f"[Latência por etapa (s): "
                f"pesquisa={timings.get('research_s', 0):.2f}, "
                f"análise={timings.get('analysis_s', 0):.2f}, "
                f"relatório={timings.get('report_s', 0):.2f}, "
                f"total={timings.get('total_s', 0):.2f}]"
            )

        guardrails = result.get("guardrails", {})
        context = result.get("context_engineering", {})
        if guardrails:
            print(
                f"[Guardrails: entrada={'ok' if guardrails.get('input', {}).get('allowed', True) else 'bloqueada'}, "
                f"segredos redigidos={guardrails.get('output_redactions', 0)}]"
            )
        if context:
            notes = context.get("notes", {})
            print(
                f"[Engenharia de contexto: {notes.get('total_points', 0)} notas estruturadas, "
                f"{context.get('chars_saved', 0)} caracteres economizados]"
            )
    except GuardrailError as exc:
        print(f"\n[BLOQUEADO POR GUARDRAIL] {exc}", file=sys.stderr)
        sys.exit(2)
    except ValueError as exc:
        print(f"\n[ERRO de configuração] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERRO] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if agent is not None:
            agent.close()


if __name__ == "__main__":
    main()
