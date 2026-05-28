# Caso de uso: Agente de Pesquisa e Relatório

## Objetivo (fase atual)

Implementar o pipeline canônico e o **baseline vanilla** — referência mínima de latência, controle de fluxo e número de chamadas API antes de comparar frameworks.

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

## Variáveis controladas

| Variável    | Fonte                                                                   |
|-------------|-------------------------------------------------------------------------|
| Prompts     | `common/common/research_prompts.py` (`*_system` + `*_user` por etapa)  |
| Modelo      | `OPENAI_MODEL`                                                          |
| Temperatura | `OPENAI_TEMPERATURE`                                                    |
| Tópico      | `--topic` na CLI                                                        |

## Próximas fases (fora do escopo atual)

Comparação com LangChain, LangGraph, CrewAI e OpenAI Agents SDK usando o mesmo caso de uso e os mesmos prompts compartilhados em `common/`.
