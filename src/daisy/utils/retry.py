"""
Retry decorators and utilities.
"""

import time
import random
import logging
from typing import Type, Tuple, Union, Optional
from selenium.common.exceptions import WebDriverException

# OpenAI API exceptions
try:
    from openai._exceptions import (
        APIError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
    )
    OPENAI_EXCEPTIONS = (APIError, APITimeoutError, RateLimitError, InternalServerError)
except ImportError:
    OPENAI_EXCEPTIONS = ()

# Common network/connection exceptions
NETWORK_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# JSON parsing exceptions
try:
    from json import JSONDecodeError
    JSON_EXCEPTIONS = (JSONDecodeError,)
except ImportError:
    JSON_EXCEPTIONS = ()


def exponential_backoff_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    logger: Optional[logging.Logger] = None,
    operation_name: str = "operation"
):
    """
    Generic decorator for exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for first retry
        max_delay: Maximum delay in seconds
        backoff_factor: Factor to multiply delay by for each retry
        exceptions: Tuple of exception types to retry on. If None, uses default retryable exceptions
        logger: Optional logger for retry messages
        operation_name: Name of the operation for logging purposes
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Use provided exceptions or default retryable exceptions
            retryable_exceptions = exceptions or (
                WebDriverException,
                *OPENAI_EXCEPTIONS,
                *NETWORK_EXCEPTIONS,
                *JSON_EXCEPTIONS,
            )
            
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        error_msg = f"{operation_name} failed after {max_retries + 1} attempts: {str(e)}"
                        if logger:
                            logger.error(error_msg)
                        else:
                            print(error_msg)
                        raise e

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor**attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    delay *= 0.5 + random.random() * 0.5

                    retry_msg = f"{operation_name} attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f} seconds..."
                    if logger:
                        logger.warning(retry_msg)
                    else:
                        print(retry_msg)
                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception, raise immediately
                    raise e

            # This should never be reached
            raise last_exception

        return wrapper

    return decorator


def llm_api_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    logger: Optional[logging.Logger] = None
):
    """
    Specialized retry decorator for LLM API calls.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for first retry
        max_delay: Maximum delay in seconds
        backoff_factor: Factor to multiply delay by for each retry
        logger: Optional logger for retry messages
    """
    return exponential_backoff_retry(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        exceptions=(*OPENAI_EXCEPTIONS, *NETWORK_EXCEPTIONS, *JSON_EXCEPTIONS),
        logger=logger,
        operation_name="LLM API call"
    )


def webdriver_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    logger: Optional[logging.Logger] = None
):
    """
    Specialized retry decorator for WebDriver operations.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for first retry
        max_delay: Maximum delay in seconds
        backoff_factor: Factor to multiply delay by for each retry
        logger: Optional logger for retry messages
    """
    return exponential_backoff_retry(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        exceptions=(WebDriverException,),
        logger=logger,
        operation_name="WebDriver operation"
    )
