#!/usr/bin/env bash
# setup.sh — Configuração do ambiente para macOS / Linux
set -e

echo "==> Criando ambiente virtual Python..."
python3 -m venv .venv

echo "==> Ativando ambiente virtual..."
source .venv/bin/activate

echo "==> Instalando dependências base..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "==> Instalando pacotes do projeto..."
pip install -e ./common -e ./vanilla -e ./langgraph_pipeline -e ./crewai_pipeline -q

echo "==> Configurando arquivo de ambiente..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "    Arquivo .env criado a partir de .env.example"
    echo "    Edite .env e preencha sua OPENAI_API_KEY antes de executar."
else
    echo "    Arquivo .env já existe — mantido sem alterações."
fi

echo ""
echo "Setup concluído!"
echo ""
echo "Próximos passos:"
echo "  1. source .venv/bin/activate"
echo "  2. Edite .env e defina OPENAI_API_KEY=sk-..."
echo "  3. start_vanilla --topic \"Seu tópico de pesquisa\""
echo "  4. start_langgraph --topic \"Seu tópico de pesquisa\""
echo "  5. start_crewai --topic \"Seu tópico de pesquisa\""
