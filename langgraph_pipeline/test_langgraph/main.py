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
    parser.add_argument(
        "--durable",
        action="store_true",
        help="Ativa estados duráveis (checkpointing do estado a cada nó)",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Identifica a thread durável (gerado automaticamente se omitido)",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=None,
        help="Caminho de banco SQLite para durabilidade entre processos (opcional)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma uma thread durável existente (requer --durable e --thread-id)",
    )
    args = parser.parse_args()

    if args.resume and (not args.durable or not args.thread_id):
        print(
            "\n[ERRO de uso] --resume requer --durable e --thread-id.",
            file=sys.stderr,
        )
        sys.exit(1)

    agent = None
    try:
        agent = LangGraphResearchReportAgent(
            durable=args.durable, checkpoint_db=args.checkpoint_db
        )
        if args.resume:
            result = agent.resume_research_pipeline(args.thread_id)
        else:
            result = agent.run_research_pipeline(args.topic, thread_id=args.thread_id)

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

        durable = result.get("durable", {})
        if durable.get("enabled"):
            print(
                f"[Estado durável: thread={durable.get('thread_id')}, "
                f"backend={durable.get('checkpointer')}, "
                f"checkpoints={durable.get('checkpoints', 0)}]"
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
