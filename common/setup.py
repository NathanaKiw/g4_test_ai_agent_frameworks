from setuptools import find_packages, setup

setup(
    name="common",
    version="0.2.0",
    description="Prompts e utilitários compartilhados — Agente de Pesquisa e Relatório",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pymongo>=4.6.0",
    ],
)
