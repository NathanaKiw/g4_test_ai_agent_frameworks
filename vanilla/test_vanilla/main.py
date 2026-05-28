"""CLI do baseline vanilla."""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from .research_agent import ResearchReportAgent  # noqa: E402 (após load_dotenv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente de Pesquisa — baseline Vanilla")
    parser.add_argument("--topic", required=True, help="Tópico de pesquisa")
    args = parser.parse_args()

    agent = None
    try:
        agent = ResearchReportAgent()
        result = agent.run_research_pipeline(args.topic)

        print("\n=== RELATÓRIO FINAL (Vanilla) ===\n")
        print(result["report"])

        api_calls = result.get("api_calls", 3)
        timings = result.get("stage_timings", {})
        print(f"\n[Chamadas API: {api_calls}]")
        if timings:
            print(
                f"[Latência por etapa (s): "
                f"pesquisa={timings.get('research_s', 0):.2f}, "
                f"análise={timings.get('analysis_s', 0):.2f}, "
                f"relatório={timings.get('report_s', 0):.2f}, "
                f"total={timings.get('total_s', 0):.2f}]"
            )
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
