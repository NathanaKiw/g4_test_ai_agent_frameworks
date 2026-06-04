"""Stub mínimo para `langchain_openai.ChatOpenAI` usado nos testes."""
from typing import Sequence


class _Response:
    def __init__(self, content: str):
        self.content = content


class ChatOpenAI:
    def __init__(self, model=None, temperature=0.0, api_key=None, base_url=None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url

    def invoke(self, messages: Sequence[object]):
        # Return the last message content as the response content.
        if not messages:
            return _Response("")
        last = messages[-1]
        content = getattr(last, "content", str(last))
        return _Response(content)
