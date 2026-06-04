# Agente de Pesquisa e Relatório — Comparativo de Frameworks LLM

Projeto de **benchmark comparativo de frameworks de agentes LLM**, implementando um pipeline canônico idêntico em múltiplos frameworks e medindo métricas objetivas de desempenho.

## Objetivo

Comparar como diferentes frameworks de orquestração de agentes (LangChain, LangGraph, CrewAI, OpenAI Agents SDK) se comportam ao executar o **mesmo pipeline** com os **mesmos prompts**, medindo:

- `api_calls` — número total de tentativas de chamada à API (inclui retries)
- `stage_timings` — latência por etapa (`research_s`, `analysis_s`, `report_s`)

A fase atual entrega o **baseline Vanilla** e dois protótipos mínimos da Sprint 2: **LangGraph** e **CrewAI**.

## Pipeline canônico

```
Tópico → [Pesquisador] → [Analista] → [Redator] → Relatório Executivo
```

| Etapa | Papel       | Objetivo                                              |
|-------|-------------|-------------------------------------------------------|
| 1     | Pesquisador | Coletar fatos, contexto e tendências sobre o tópico   |
| 2     | Analista    | Interpretar dados, riscos, cenários e lacunas         |
| 3     | Redator     | Produzir relatório executivo final                    |

Os prompts compartilhados estão em `common/common/research_prompts.py` e são reutilizados por todos os frameworks.

## Estrutura do repositório

```
g4_test_ai_agent_frameworks/
├── common/                     # Pacote compartilhado (prompts, config, logging, MongoDB)
│   └── common/
│       ├── config.py
│       ├── logging_config.py
│       ├── research_prompts.py
│       └── mongodb/
│           └── research_data.py
├── vanilla/                    # Baseline — API Groq direta (sem framework)
│   └── test_vanilla/
│       ├── config.py
│       ├── research_agent.py
│       └── main.py
├── langgraph_pipeline/         # Protótipo Sprint 2 — LangGraph StateGraph
│   └── test_langgraph/
│       ├── config.py
│       ├── research_agent.py
│       └── main.py
├── crewai_pipeline/            # Protótipo Sprint 2 — CrewAI sequencial
│   └── test_crewai/
│       ├── config.py
│       ├── research_agent.py
│       └── main.py
├── docs/
│   ├── CASO_DE_USO.md
│   └── COMO_RODAR.md
├── experiments/
│   └── benchmark_pipelines.py  # Benchmark real que gera métricas e gráficos
├── requirements.txt
├── setup.sh                    # Setup para macOS / Linux
├── setup.ps1                   # Setup para Windows
└── .env.example
```

## Dependências

- Python >= 3.10
- [openai](https://pypi.org/project/openai/) >= 1.40.0
- [python-dotenv](https://pypi.org/project/python-dotenv/) >= 1.0.0
- [pymongo](https://pypi.org/project/pymongo/) >= 4.6.0 *(opcional — persistência)*
- [tenacity](https://pypi.org/project/tenacity/) >= 8.2.0 *(retry automático)*
- [langgraph](https://pypi.org/project/langgraph/) >= 1.0.0
- [langchain-openai](https://pypi.org/project/langchain-openai/) >= 0.3.0
- [crewai](https://pypi.org/project/crewai/) >= 1.0.0
- [Pillow](https://pypi.org/project/Pillow/) >= 10.0.0 *(geração dos gráficos PNG)*

## Instalação

### macOS / Linux

```bash
# 1. Clone o repositório e entre na pasta
git clone <url-do-repo>
cd g4_test_ai_agent_frameworks

# 2. Execute o script de setup
chmod +x setup.sh
./setup.sh

# 3. Ative o ambiente virtual
source .venv/bin/activate

# 4. Configure sua chave de API
cp .env.example .env
# Edite .env e preencha GROQ_API_KEY
```

### Windows (PowerShell)

```powershell
# 1. Clone o repositório e entre na pasta
git clone <url-do-repo>
cd g4_test_ai_agent_frameworks

# 2. Execute o script de setup
.\setup.ps1

# 3. Ative o ambiente virtual
.\.venv\Scripts\activate

# 4. Configure sua chave de API
copy .env.example .env
# Edite .env e preencha GROQ_API_KEY
```

### Instalação manual (qualquer SO)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# ou: .\.venv\Scripts\activate   # Windows

pip install -r requirements.txt
pip install -e ./common -e ./vanilla -e ./langgraph_pipeline -e ./crewai_pipeline
cp .env.example .env
```

## Configuração

Edite o arquivo `.env` gerado:

```dotenv
GROQ_API_KEY=gsk-...            # Obrigatório
GROQ_MODEL=llama-3.3-70b-versatile  # Padrão: llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.0            # Padrão: 0.0
GROQ_BASE_URL=https://api.groq.com/openai/v1
# MONGODB_URI=mongodb://localhost:27017/ # Opcional; descomente para persistir no MongoDB
```

## Execução

```bash
start_vanilla --topic "Impacto da IA na educação brasileira"
start_langgraph --topic "Impacto da IA na educação brasileira"
start_crewai --topic "Impacto da IA na educação brasileira"
```

## Testes

```bash
python -m unittest discover -s tests
```

## Experimento com gráficos

Além dos comandos individuais dos agentes, o projeto inclui um experimento em
`experiments/benchmark_pipelines.py` para comparar visualmente Vanilla,
LangGraph e CrewAI. No modo padrão, ele executa os pipelines reais, coleta as
métricas retornadas por cada implementação e gera tabelas, relatório, dashboard
HTML e gráficos PNG.

```bash
python experiments/benchmark_pipelines.py --runs 1 --topic "Impacto da IA na educação brasileira"
```

Esse comando faz chamadas reais à API configurada no `.env` e consome quota.
Use `--output-dir caminho/da/pasta` para salvar os artefatos em outro local.

O comando gera os artefatos em `artifacts/benchmark/`:

| Arquivo | Descrição |
|---------|-----------|
| `benchmark_dashboard.html` | Dashboard visual com cards, tabela resumida e gráficos incorporados |
| `benchmark_table.md` | Tabela Markdown pronta para anexar ao relatório |
| `benchmark_results.csv` | Dados por execução e linhas de resumo |
| `benchmark_report.md` | Resumo textual do benchmark |
| `benchmark_manifest.json` | Manifesto da execução com parâmetros, frameworks e arquivos gerados |
| `avg_total_time.png` | Gráfico de barras com tempo médio total por framework |
| `avg_stage_time.png` | Gráfico agrupado com médias de pesquisa, análise e relatório |
| `stage_side_by_side.png` | Comparação lado a lado das três etapas por framework |

Os gráficos são renderizados com `Pillow`, sem depender de interface gráfica.
O dashboard HTML referencia os PNGs gerados na mesma pasta, então pode ser
aberto diretamente no navegador.

### Saída esperada

```
=== RELATÓRIO FINAL (<framework>) ===

# Relatório Executivo: Impacto da IA na Educação Brasileira

## Resumo Executivo
...

## Introdução
...

[Chamadas API: 3]
[Latência por etapa (s): pesquisa=8.32, análise=6.51, relatório=9.14]
```

## Variáveis de ambiente

| Variável             | Obrigatória | Descrição                         | Padrão                   |
|----------------------|-------------|-----------------------------------|--------------------------|
| `GROQ_API_KEY`       | Sim         | Chave de acesso à API Groq        | —                        |
| `GROQ_MODEL`         | Não         | Modelo a ser utilizado            | `llama-3.3-70b-versatile`|
| `GROQ_TEMPERATURE`   | Não         | Temperatura de geração            | `0.0`                    |
| `GROQ_BASE_URL`      | Não         | Endpoint OpenAI-compatible da Groq | `https://api.groq.com/openai/v1` |
| `MONGODB_URI`        | Não         | URI de conexão MongoDB            | desativado                  |

## Métricas de benchmark

Cada execução retorna:

```json
{
  "framework": "Vanilla (Groq API) | LangGraph | CrewAI",
  "api_calls": 3,
  "stage_timings": {
    "research_s": 8.32,
    "analysis_s": 6.51,
    "report_s": 9.14
  },
  "topic": "...",
  "report": "..."
}
```

Sem retries, o baseline executa três chamadas lógicas. Se houver rate limit, falha de conexão ou erro 5xx transitório, `api_calls` pode ser maior porque contabiliza as tentativas reais enviadas à API.

Esses valores serão usados como **linha de base** para comparação com os demais frameworks.

## Próximas fases

| Framework           | Status        |
|---------------------|---------------|
| Vanilla (Groq)      | Concluído         |
| LangChain           | Planejado         |
| LangGraph           | Protótipo mínimo  |
| CrewAI              | Protótipo mínimo  |
| OpenAI Agents SDK   | Planejado         |

## Licença

MIT
