"""Prompts compartilhados para o caso de uso Agente de Pesquisa e Relatório.

Separação semântica OpenAI:
- system: persona, tom e restrições de comportamento
- user: tarefa, dados de entrada e formato de saída
"""

from typing import Dict, Tuple


def researcher_system() -> str:
    """Persona e restrições — mensagem system."""
    return (
        "Você é um pesquisador especializado em síntese de informações confiáveis. "
        "Seja objetivo, factual e responda em português. "
        "Limite-se a conteúdo que um analista possa validar. "
        "Não invente URLs."
    )


def researcher_user(topic: str) -> str:
    """Tarefa da etapa 1 — mensagem user."""
    return f"""Pesquise de forma abrangente o tópico: "{topic}".

Produza um relatório de pesquisa estruturado contendo:
- Contexto e definição do tema
- Principais fatos, dados e tendências (com anos quando relevante)
- Atores, tecnologias ou fatores críticos envolvidos
- Desafios e oportunidades identificados
- Referências conceituais (sem inventar URLs)"""


def analyst_system() -> str:
    """Persona e restrições — mensagem system."""
    return (
        "Você é um analista sênior especializado em interpretação de pesquisas. "
        "Responda em português, de forma clara e estruturada. "
        "Responda apenas com a análise solicitada."
    )


def analyst_user(topic: str, research_data: str) -> str:
    """Tarefa da etapa 2 — mensagem user."""
    return f"""Analise os dados de pesquisa sobre "{topic}" e produza insights acionáveis.

Dados da pesquisa:
{research_data}

Sua análise deve incluir:
- Padrões e tendências principais
- Riscos e implicações (curto e médio prazo)
- Comparativo de cenários (otimista vs conservador)
- Lacunas de informação e recomendações de aprofundamento
- Conclusões preliminares para tomada de decisão"""


def report_writer_system() -> str:
    """Persona e restrições — mensagem system."""
    return (
        "Você é um redator de relatórios executivos para audiência técnica e de negócios. "
        "Use tom profissional, em português, com markdown leve (títulos e listas). "
        "Responda apenas com o relatório solicitado."
    )


def report_writer_user(topic: str, analysis_data: str) -> str:
    """Tarefa da etapa 3 — mensagem user."""
    return f"""Com base na análise sobre "{topic}", redija um relatório final profissional.

Análise disponível:
{analysis_data}

O relatório deve conter:
1. Resumo executivo (5-8 linhas)
2. Introdução ao tema
3. Principais achados
4. Análise e implicações
5. Recomendações práticas (bullet points)
6. Conclusão"""


def researcher_messages(topic: str) -> Tuple[str, str]:
    """Par (system, user) para a etapa de pesquisa."""
    return researcher_system(), researcher_user(topic)


def analyst_messages(topic: str, research_data: str) -> Tuple[str, str]:
    """Par (system, user) para a etapa de análise."""
    return analyst_system(), analyst_user(topic, research_data)


def report_writer_messages(topic: str, analysis_data: str) -> Tuple[str, str]:
    """Par (system, user) para a etapa de relatório."""
    return report_writer_system(), report_writer_user(topic, analysis_data)


def build_result(topic: str, research: str, analysis: str, report: str, timestamp: str) -> Dict:
    """Estrutura padronizada de saída para todos os frameworks."""
    return {
        "topic": topic,
        "research": research,
        "analysis": analysis,
        "report": report,
        "timestamp": timestamp,
    }
