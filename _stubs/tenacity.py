"""Stub bem simples do `tenacity` usado apenas para testes locais.

Os decoradores retornam a função inalterada e helpers são no-ops.
"""
def retry(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator


def before_sleep_log(logger, level):
    return None


def retry_if_exception(fn):
    return lambda e: False


def stop_after_attempt(n):
    return None


def wait_exponential(**kwargs):
    return None
