

import time
import functools


def retry_on_timeout(max_retries=3, backoff=2.0, exceptions=(TimeoutError, ConnectionError)):
    """Retry decorator for transient API failures (e.g. PubMed rate limits)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    wait = backoff ** attempt
                    time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator
