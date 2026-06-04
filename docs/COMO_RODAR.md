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
Ran 7 tests
OK
```

## 7. Experimento / demo com gráficos

Para gerar métricas quantitativas e gráficos PNG dos três pipelines, execute:

```powershell
python experiments/benchmark_pipelines.py --runs 10 --delay 0.02 --jitter 0.005
```

Artefatos gerados em `artifacts/benchmark/`:

- `benchmark_dashboard.html` com o resumo visual pronto para apresentação
- `benchmark_table.md` com a tabela pronta para colar no relatório
- `benchmark_results.csv` com os valores por execução
- `benchmark_report.md` com o resumo consolidado
- `avg_total_time.png` com o tempo médio total por framework
- `avg_stage_time.png` com o comparativo visual das três etapas
- `stage_side_by_side.png` com as três etapas lado a lado por framework

Se quiser uma demo mais bonita para mostrar, abra `benchmark_dashboard.html` no navegador. O arquivo já traz cards, tabela resumida e os gráficos incorporados.

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
| `start_langgraph` | `langgraph_pipeline/test_langgraph/research_agent.py` | Fluxo com `StateGraph` e nós explícitos |
| `start_crewai` | `crewai_pipeline/test_crewai/research_agent.py` | Três agentes especializados em processo sequencial |

Todos reutilizam os prompts compartilhados em `common/common/research_prompts.py`.
