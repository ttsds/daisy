"""
Constants and configuration for the Daisy pipeline.
"""

# Default LLM configurations
DEFAULT_SEARCH_LLM = "google/gemini-2.5-pro:online"
DEFAULT_PARSE_LLM = "x-ai/grok-4-fast"

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # Base delay in seconds
MAX_DELAY = 60.0  # Maximum delay in seconds
BACKOFF_FACTOR = 2.0  # Exponential backoff factor
JITTER = True  # Add random jitter to prevent thundering herd
