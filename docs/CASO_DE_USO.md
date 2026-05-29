# Caso de uso: Agente de Pesquisa e Relatório

## Objetivo (fase atual)

Implementar o pipeline canônico, o **baseline vanilla** e os protótipos mínimos de Sprint 2 em **LangGraph** e **CrewAI**.

## Fluxo canônico

```mermaid
flowchart LR
    A[Tópico] --> B[Pesquisador]
    B --> C[Analista]
    C --> D[Redator]
    D --> E[Relatório executivo]
```

## Baseline (Vanilla)

- Sem dependências de orquestração
- Três chamadas lógicas explícitas: `client.chat.completions.create`
- `api_calls = 3` sem retries; pode aumentar se houver tentativas automáticas em erros transitórios
- `stage_timings`: `research_s`, `analysis_s`, `report_s`, `total_s`

Implementação: `vanilla/test_vanilla/research_agent.py`

## Protótipos Sprint 2

| Framework | Implementação | Foco |
|-----------|---------------|------|
| LangGraph | `langgraph_pipeline/test_langgraph/research_agent.py` | Controle explícito do fluxo com `StateGraph` |
| CrewAI | `crewai_pipeline/test_crewai/research_agent.py` | Colaboração sequencial entre agentes especializados |

Comandos:

```bash
start_vanilla --topic "Impacto da IA na educação brasileira"
start_langgraph --topic "Impacto da IA na educação brasileira"
start_crewai --topic "Impacto da IA na educação brasileira"
```

## Variáveis controladas

| Variável    | Fonte                                                                   |
|-------------|-------------------------------------------------------------------------|
| Prompts     | `common/common/research_prompts.py` (`*_system` + `*_user` por etapa)  |
| Modelo      | `GROQ_MODEL`                                                            |
| Temperatura | `GROQ_TEMPERATURE`                                                      |
| Tópico      | `--topic` na CLI                                                        |

## Próximas fases

Evoluir a comparação com OpenAI Agents SDK, guardrails, engenharia de contexto e consolidação das métricas de latência, tokens e custo.
