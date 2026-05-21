# Agente de Pesquisa e Relatório — Baseline Vanilla

Estabelecimento do **caso de uso base** e da implementação **vanilla** (chamadas diretas à API OpenAI), servindo como **linha de base** para comparação futura com LangChain, LangGraph, CrewAI e OpenAI Agents SDK.

## Caso de uso

Pipeline sequencial em três etapas:

| Etapa | Papel | Objetivo |
|-------|--------|----------|
| 1 | Pesquisador | Coletar fatos, contexto e tendências sobre o tópico |
| 2 | Analista | Interpretar dados, riscos, cenários e lacunas |
| 3 | Redator | Produzir relatório executivo final |

Os prompts compartilhados ficam em `common/common/research_prompts.py`.

## Baseline Vanilla

Em `vanilla/` não há framework de orquestração: o fluxo é controlado manualmente com **três** chamadas `chat.completions` via SDK OpenAI.

Métricas expostas na saída:

- `api_calls` — sempre 3 (referência do experimento)
- `stage_timings` — latência por etapa (`research_s`, `analysis_s`, `report_s`)

## Estrutura

```
g4_test_ai_agent_frameworks/
├── common/           # Prompts, config, logging, MongoDB opcional
├── vanilla/          # Baseline — API OpenAI direta
├── docs/CASO_DE_USO.md
├── requirements.txt
├── setup.ps1
└── .env.example
```

## Instalação

```powershell
cd g4_test_ai_agent_frameworks
.\setup.ps1
.\.venv\Scripts\activate
# Edite .env e defina OPENAI_API_KEY
```

Ou manualmente:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e ./common -e ./vanilla
copy .env.example .env
```

## Execução

```powershell
start_vanilla --topic "Impacto da IA na educação brasileira"
```

## Variáveis de ambiente

| Variável | Descrição |
|----------|-----------|
| `OPENAI_API_KEY` | Obrigatória |
| `OPENAI_MODEL` | Padrão: `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | Padrão: `0.0` |
| `RESEARCH_TOPIC` | Tópico padrão (opcional) |
| `MONGODB_URI` | Persistência opcional |

## Licença

MIT
