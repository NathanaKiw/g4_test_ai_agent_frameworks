from importlib import import_module
import unittest.mock as mock

ra = import_module('test_vanilla.research_agent')

class FakeRateLimitError(Exception):
    pass

exc = FakeRateLimitError()
exc.body = {'error': {'code': 'insufficient_quota'}}

with mock.patch('test_vanilla.research_agent.RateLimitError', FakeRateLimitError):
    print('RateLimitError in module:', ra.RateLimitError)
    print('isinstance?', isinstance(exc, ra.RateLimitError))
    print('error_code via _error_code:', ra._error_code(exc) if hasattr(ra, '_error_code') else 'no _error_code')
    print('predicate result:', ra._is_retryable_openai_error(exc))
