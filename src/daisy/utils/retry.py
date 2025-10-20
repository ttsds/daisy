"""
Retry decorators and utilities.
"""

import time
import random
from selenium.common.exceptions import WebDriverException


def exponential_backoff_retry(
    max_retries=3, base_delay=1.0, max_delay=60.0, backoff_factor=2.0
):
    """Decorator for exponential backoff retry logic"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except WebDriverException as e:
                    last_exception = e
                    if attempt == max_retries:
                        print(
                            f"WebDriver operation failed after {max_retries + 1} attempts: {str(e)}"
                        )
                        raise e

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor**attempt), max_delay)

                    # Add jitter
                    delay *= 0.5 + random.random() * 0.5

                    print(
                        f"WebDriver operation attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception, raise immediately
                    raise e

            # This should never be reached
            raise last_exception

        return wrapper

    return decorator
