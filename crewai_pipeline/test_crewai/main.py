"""CLI do protótipo CrewAI."""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

from .research_agent import CrewAIResearchReportAgent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente de Pesquisa — CrewAI")
    parser.add_argument("--topic", required=True, help="Tópico de pesquisa")
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Ativa o modo hierárquico com delegação autônoma (agente gerente)",
    )
    args = parser.parse_args()

    agent = None
    try:
        agent = CrewAIResearchReportAgent(hierarchical=args.hierarchical)
        result = agent.run_research_pipeline(args.topic)

        modo = "hierárquico/delegação" if args.hierarchical else "sequencial"
        print(f"\n=== RELATÓRIO FINAL (CrewAI — {modo}) ===\n")
        print(result["report"])

        timings = result.get("stage_timings", {})
        print(f"\n[Chamadas/Tarefas LLM: {result.get('api_calls', 3)}]")
        if timings:
            print(
                f"[Latência por etapa (s): "
                f"pesquisa={timings.get('research_s', 0):.2f}, "
                f"análise={timings.get('analysis_s', 0):.2f}, "
                f"relatório={timings.get('report_s', 0):.2f}, "
                f"total={timings.get('total_s', 0):.2f}]"
            )

        delegation = result.get("delegation", {})
        if delegation:
            print(
                f"[Orquestração: processo={delegation.get('process', 'sequential')}, "
                f"delegação autônoma={'sim' if delegation.get('enabled') else 'não'}]"
            )
    except ValueError as exc:
        print(f"\n[ERRO de configuração] {exc}", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"\n[ERRO de dependência] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERRO] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if agent is not None:
            agent.close()


if __name__ == "__main__":
    main()
