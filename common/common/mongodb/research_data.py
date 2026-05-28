"""Persistência opcional de resultados de pesquisa (MongoDB)."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger

logger = get_logger("research_data")


class ResearchDataService:
    """Serviço para armazenar resultados do agente de pesquisa."""

    def __init__(
        self,
        connection_string: str | None = None,
    ):
        connection_string = connection_string or os.getenv("MONGODB_URI")
        self._enabled = False
        self.client = None
        self.collection = None
        self.logger = get_logger("research_data_service")

        if not connection_string:
            self.logger.info("MongoDB não configurado — persistência desativada")
            return

        try:
            from pymongo import MongoClient

            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=2000)
            self.client.admin.command("ping")
            self.db = self.client["research_agent_db"]
            self.collection = self.db["research_reports"]
            self._enabled = True
            self.logger.info("MongoDB conectado para persistência de relatórios")
        except Exception as exc:
            self.logger.warning("MongoDB indisponível — persistência desativada: %s", exc)

    def insert_research_report(self, report_data: Dict[str, Any]) -> Optional[str]:
        if not self._enabled or self.collection is None:
            return None

        document = {**report_data, "stored_at": datetime.now()}
        result = self.collection.insert_one(document)
        self.logger.info("Relatório armazenado para tópico: %s", document.get("topic", "?"))
        return str(result.inserted_id)

    def get_report_by_topic(self, topic: str) -> Optional[Dict]:
        if not self._enabled or self.collection is None:
            return None
        return self.collection.find_one({"topic": topic})

    def get_all_topics(self) -> List[str]:
        if not self._enabled or self.collection is None:
            return []
        return self.collection.distinct("topic")

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.logger.info("Conexão MongoDB encerrada")
