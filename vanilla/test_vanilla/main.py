"""CLI do baseline vanilla."""

import argparse

from dotenv import load_dotenv

from .research_agent import ResearchReportAgent

load_dotenv()


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
        print(f"\n[Chamadas API: {result.get('api_calls', 3)}]")
        timings = result.get("stage_timings")
        if timings:
            print(
                f"[Latência por etapa (s): pesquisa={timings['research_s']:.2f}, "
                f"análise={timings['analysis_s']:.2f}, relatório={timings['report_s']:.2f}]"
            )
    finally:
        if agent is not None:
            agent.close()


if __name__ == "__main__":
    main()
