# Como Rodar o Projeto

Este guia mostra como instalar, configurar e executar os protótipos do projeto
comparativo de frameworks de agentes LLM.

## 1. Entrar na pasta do projeto

```bash
cd caminho/para/g4_test_ai_agent_frameworks
```

## 2. Criar e ativar o ambiente virtual

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

## 3. Instalar dependências

```bash
pip install -r requirements.txt
pip install -e ./common -e ./vanilla -e ./langgraph_pipeline -e ./crewai_pipeline
```

Também é possível usar o script de setup no macOS/Linux:

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

No Windows:

```powershell
.\setup.ps1
.\.venv\Scripts\activate
```

## 4. Configurar variáveis de ambiente

Crie o arquivo `.env` a partir do exemplo:

```bash
cp .env.example .env
```

No Windows:

```powershell
copy .env.example .env
```

Edite o `.env` e preencha sua chave da Groq:

```dotenv
GROQ_API_KEY=gsk-sua-chave-aqui
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.0
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

MongoDB é opcional. Para persistir relatórios, descomente e ajuste:

```dotenv
MONGODB_URI=mongodb://localhost:27017/
```

## 5. Rodar os protótipos

Baseline vanilla:

```bash
start_vanilla --topic "Impacto da IA na educação brasileira"
```

Protótipo LangGraph:

```bash
start_langgraph --topic "Impacto da IA na educação brasileira"
```

Protótipo CrewAI:

```bash
start_crewai --topic "Impacto da IA na educação brasileira"
```

### Modos avançados (opcionais)

Esses modos são desligados por padrão e **não** afetam o benchmark.

LangGraph com estados duráveis (checkpointing + retomada):

```bash
# Executa e persiste o estado sob um thread_id
start_langgraph --topic "Impacto da IA na educação" --durable --thread-id exec-001

# Retoma a mesma thread sem reprocessar etapas já concluídas
start_langgraph --topic "Impacto da IA na educação" --durable --thread-id exec-001 --resume
```

CrewAI com delegação autônoma (processo hierárquico):

```bash
start_crewai --topic "Impacto da IA na educação" --hierarchical
```

## 6. Rodar os testes

Os testes não fazem chamadas reais à API.

Use um destes comandos conforme seu sistema:

- macOS / Linux:

```bash
PYTHONPATH=common:vanilla:langgraph_pipeline:crewai_pipeline python3 -m unittest discover -s tests
```

- Windows PowerShell (venv ativado):

```powershell
$env:PYTHONPATH = "common;vanilla;langgraph_pipeline;crewai_pipeline"
python -m unittest discover -s tests
```

Resultado esperado:

```text
Ran 25 tests
OK
```

## 7. Experimento / benchmark com gráficos

O projeto também possui um experimento para comparar visualmente os três
pipelines: Vanilla, LangGraph e CrewAI. Por padrão, ele executa as
implementações reais dos agentes, coleta as métricas retornadas por cada
pipeline (`api_calls`, `research_s`, `analysis_s`, `report_s` e `total_s`) e
gera gráficos a partir desses dados.

Este modo faz chamadas reais à API configurada no `.env`, então requer uma
`GROQ_API_KEY` válida e consome quota/créditos da conta.

Execute o comando abaixo com o ambiente virtual ativado:

macOS / Linux:

```bash
python experiments/benchmark_pipelines.py --runs 1 --topic "Impacto da IA na educação brasileira"
```

Windows PowerShell:

```powershell
python experiments/benchmark_pipelines.py --runs 1 --topic "Impacto da IA na educação brasileira"
```

O que esse comando faz:

- executa Vanilla, LangGraph e CrewAI usando a API real;
- coleta os tempos medidos por cada implementação;
- calcula médias de tempo total e tempo por etapa;
- gera tabelas, CSV, relatório Markdown, dashboard HTML e gráficos PNG.

Parâmetros principais:

| Parâmetro | Descrição |
|-----------|-----------|
| `--runs 1` | Define quantas execuções serão feitas para cada framework |
| `--topic "..."` | Altera o tópico usado como entrada do benchmark |
| `--output-dir artifacts/benchmark` | Define a pasta onde os arquivos serão salvos |

Depois da execução, os arquivos ficam em `artifacts/benchmark/`:

| Arquivo | Finalidade |
|---------|------------|
| `benchmark_dashboard.html` | Dashboard pronto para apresentar, com cards, tabela e gráficos |
| `benchmark_table.md` | Tabela em Markdown para copiar para o relatório do trabalho |
| `benchmark_results.csv` | Dados completos de cada execução e linhas de resumo |
| `benchmark_report.md` | Resumo textual consolidado do experimento |
| `benchmark_manifest.json` | Registro da execução com parâmetros, frameworks e artefatos gerados |
| `avg_total_time.png` | Gráfico do tempo médio total por framework |
| `avg_stage_time.png` | Gráfico comparando pesquisa, análise e relatório |
| `stage_side_by_side.png` | Gráfico das etapas lado a lado por framework |

Para visualizar a demo, abra este arquivo no navegador:

```text
artifacts/benchmark/benchmark_dashboard.html
```

Se aparecer o erro abaixo, reinstale as dependências porque o gerador dos
gráficos usa a biblioteca `Pillow`:

```text
ModuleNotFoundError: No module named 'PIL'
```

```bash
pip install -r requirements.txt
```

## 8. Erros comuns

### GROQ_API_KEY ausente

Se aparecer:

```text
GROQ_API_KEY é obrigatório
```

verifique se o arquivo `.env` existe e se contém uma chave Groq válida.

### Quota insuficiente

Se aparecer:

```text
Error code: 429
code: insufficient_quota
```

a chave chegou à API, mas a conta/projeto Groq está sem créditos ou billing
ativo. Verifique o billing da conta na plataforma da Groq antes de rodar
novamente.

## 9. O que cada comando executa

| Comando | Implementação | Descrição |
|---------|---------------|-----------|
| `start_vanilla` | `vanilla/test_vanilla/research_agent.py` | Baseline com chamadas diretas à API Groq |
| `start_langgraph` | `langgraph_pipeline/test_langgraph/research_agent.py` | Fluxo com `StateGraph`, guardrails, engenharia de contexto e modo durável (`--durable`) |
| `start_crewai` | `crewai_pipeline/test_crewai/research_agent.py` | Três agentes especializados (sequencial; `--hierarchical` ativa delegação autônoma) |

Todos reutilizam os prompts compartilhados em `common/common/research_prompts.py`.
