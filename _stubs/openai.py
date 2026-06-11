"""Stub do pacote `openai` usado nos testes locais.

Este arquivo fornece exceções e uma classe `OpenAI` mínima para evitar
ImportError durante a execução dos testes sem a dependência instalada.
"""
class APIConnectionError(Exception):
    pass


class APIStatusError(Exception):
    def __init__(self, status_code=500, body=None):
        super().__init__(f"Status {status_code}")
        self.status_code = status_code
        self.body = body


class RateLimitError(Exception):
    pass


class _ChoicesMessage:
    def __init__(self, content: str):
        class Message:
            def __init__(self, content):
                self.content = content

        self.message = Message(content)


class _CompletionResponse:
    def __init__(self, text: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class _Completions:
    def create(self, **kwargs):
        return _CompletionResponse("")


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class OpenAI:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = _Chat()
