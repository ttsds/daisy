"""
LLM-based media source collection.
"""

import os
import logging
from datetime import datetime
import uuid
import json
import random
import time
from typing import Optional

from openai import OpenAI
from openai._exceptions import (
    APIError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

from daisy.core import ListSource, MediaItem, MediaItemList
from daisy.core.constants import (
    DEFAULT_SEARCH_LLM,
    DEFAULT_PARSE_LLM,
    MAX_RETRIES,
    BASE_DELAY,
    MAX_DELAY,
    BACKOFF_FACTOR,
    JITTER,
)


class BaseLLMSource(ListSource):
    """Base class for all LLM-based sources to reduce code duplication"""

    def __init__(
        self,
        language: str,
        save_file: str,
        source_id: str,
        content_type: str,
        overwrite: bool = False,
        llm_id_search: str = DEFAULT_SEARCH_LLM,
        llm_id_parse: str = DEFAULT_PARSE_LLM,
        log_dir: Optional[str] = None,
    ):
        super().__init__(language, save_file, source_id, overwrite)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_KEY"),
        )
        self.llm_id_search = llm_id_search
        self.llm_id_parse = llm_id_parse
        self.content_type = content_type

        # Set up logging
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(save_file), f"logs_{self.source_id}"
        )
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging infrastructure for LLM interactions"""
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.source_id}")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler
        log_file = os.path.join(
            self.log_dir,
            f"{self.source_id}_{self.language.iso2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Log initialization
        self.logger.info(f"Initialized {self.__class__.__name__}")
        self.logger.info(
            f"Language: {self.language.english_name} ({self.language.native_name})"
        )
        self.logger.info(f"Content type: {self.content_type}")
        self.logger.info(f"Search LLM: {self.llm_id_search}")
        self.logger.info(f"Parse LLM: {self.llm_id_parse}")
        self.logger.info(f"Log file: {log_file}")

    def _log_llm_interaction(
        self,
        interaction_type: str,
        model: str,
        prompt: str,
        response: str,
        session_id: str,
    ):
        """Log LLM interaction details"""
        log_entry = {
            "session_id": session_id,
            "interaction_type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt": prompt,
            "response": response,
            "language": self.language.iso2,
            "content_type": self.content_type,
            "source_id": self.source_id,
        }

        # Log to file as JSON
        log_file = os.path.join(self.log_dir, f"llm_interactions_{session_id}.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Log summary to main log
        self.logger.info(
            f"LLM {interaction_type} - Model: {model}, Session: {session_id}"
        )
        self.logger.debug(f"Prompt: {prompt[:200]}...")
        self.logger.debug(f"Response: {response[:200]}...")

    def _make_llm_request(
        self, model: str, messages: list, response_format=None, session_id: str = None
    ):
        """
        Make an LLM request with retry logic for common API errors.
        """
        # Define retryable exceptions
        retryable_exceptions = (
            APIError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
            json.JSONDecodeError,
            ConnectionError,
            TimeoutError,
        )

        last_exception = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                if response_format:
                    response = self.client.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                    )

                # Log successful request
                if session_id:
                    self.logger.info(
                        f"LLM request successful on attempt {attempt + 1} for session {session_id}"
                    )

                return response

            except retryable_exceptions as e:
                last_exception = e

                if attempt == MAX_RETRIES:
                    # Last attempt failed
                    self.logger.error(
                        f"LLM request failed after {MAX_RETRIES + 1} attempts: {str(e)}"
                    )
                    raise e

                # Calculate delay with exponential backoff
                delay = min(BASE_DELAY * (BACKOFF_FACTOR**attempt), MAX_DELAY)

                # Add jitter
                if JITTER:
                    delay *= 0.5 + random.random() * 0.5

                # Log retry attempt
                self.logger.warning(
                    f"LLM request attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds..."
                )

                time.sleep(delay)

            except Exception as e:
                # Non-retryable exception, raise immediately
                self.logger.error(f"Non-retryable LLM request error: {str(e)}")
                raise e

        # This should never be reached
        raise last_exception

    def _get_search_prompt(self) -> str:
        """Generate the search prompt for the specific content type"""
        return f"""
        What are the top {self.content_type} in the {self.language.english_name} ({self.language.native_name}) language which are made by native speakers, for native speakers? List at least 50 if possible. If listing 50 is not possible, list as many as possible. Use the native names of the {self.content_type} if possible.  
        """

    def _get_parse_prompt(self, response: str) -> str:
        """Generate the parse prompt for the specific content type"""
        return f"""
        Parse the following text into a list of {self.content_type}.
        The items should have the following fields:
        - name: The name of the {self.content_type} - use the native name if possible and DO NOT include the translated name in brackets or parentheses.
        - description: The description of the {self.content_type}
        - categories: The categories of the {self.content_type}
        - language: The language of the {self.content_type}
        - country: The country of the {self.content_type}
        {response}
        """

    def _process_items(self, items: list[MediaItem]) -> list[MediaItem]:
        """Process the parsed items and add metadata"""
        processed_items = []
        for item in items:
            # Update the MediaItem with additional metadata
            item_dict = item.model_dump(mode="json")
            item_dict.update(
                {
                    "source_id": self.source_id,
                    "llm_id_search": self.llm_id_search,
                    "llm_id_parse": self.llm_id_parse,
                    "language": self.language.iso2,
                    "date_collected": datetime.now().isoformat(),
                }
            )
            # Create new MediaItem with updated data
            processed_item = MediaItem(**item_dict)
            processed_items.append(processed_item)
        return processed_items

    def search(self) -> list[MediaItem]:
        """Generic search method that works for all content types"""
        # Generate unique session ID for this search session
        session_id = str(uuid.uuid4())
        self.logger.info(f"Starting search session: {session_id}")

        try:
            # Get search results
            search_prompt = self._get_search_prompt()
            self.logger.info(f"Making search request to {self.llm_id_search}")

            response = self._make_llm_request(
                model=self.llm_id_search,
                messages=[{"role": "user", "content": search_prompt}],
                session_id=session_id,
            )
            search_response = response.choices[0].message.content

            # Log search interaction
            self._log_llm_interaction(
                "search", self.llm_id_search, search_prompt, search_response, session_id
            )

            # Parse the results
            parse_prompt = self._get_parse_prompt(search_response)
            self.logger.info(f"Making parse request to {self.llm_id_parse}")

            response = self._make_llm_request(
                model=self.llm_id_parse,
                messages=[{"role": "user", "content": parse_prompt}],
                response_format=MediaItemList,
                session_id=session_id,
            )

            parsed_response = response.choices[0].message.parsed
            items = parsed_response.items

            # Log parse interaction
            self._log_llm_interaction(
                "parse",
                self.llm_id_parse,
                parse_prompt,
                json.dumps(parsed_response.model_dump(mode="json"), ensure_ascii=False),
                session_id,
            )

            # Log results summary
            self.logger.info(f"Search completed - Found {len(items)} items")

            # Process items and add session metadata
            processed_items = self._process_items(items)
            final_items = []
            for item in processed_items:
                # Create new MediaItem with session metadata
                item_dict = item.model_dump(mode="json")
                item_dict.update(
                    {
                        "search_session_id": session_id,
                        "search_timestamp": datetime.now().isoformat(),
                    }
                )
                final_item = MediaItem(**item_dict)
                final_items.append(final_item)

            return final_items

        except Exception as e:
            self.logger.error(f"Search failed for session {session_id}: {str(e)}")
            # Log error details
            error_log = {
                "session_id": session_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat(),
                "language": self.language.iso2,
                "content_type": self.content_type,
                "source_id": self.source_id,
            }

            error_log_file = os.path.join(self.log_dir, f"errors_{session_id}.json")
            with open(error_log_file, "w", encoding="utf-8") as f:
                json.dump(error_log, f, ensure_ascii=False, indent=2)

            raise


class LLMPodcastSource(BaseLLMSource):
    """LLM-based podcast source"""

    def __init__(
        self,
        language: str,
        save_file: str,
        source_id: str,
        overwrite: bool = False,
        llm_id_search: str = DEFAULT_SEARCH_LLM,
        llm_id_parse: str = DEFAULT_PARSE_LLM,
        log_dir: Optional[str] = None,
    ):
        super().__init__(
            language,
            save_file,
            source_id,
            "podcasts",
            overwrite,
            llm_id_search,
            llm_id_parse,
            log_dir,
        )


class LLMBroadcastNewsSource(BaseLLMSource):
    """LLM-based broadcast news source"""

    def __init__(
        self,
        language: str,
        save_file: str,
        source_id: str,
        overwrite: bool = False,
        llm_id_search: str = DEFAULT_SEARCH_LLM,
        llm_id_parse: str = DEFAULT_PARSE_LLM,
        log_dir: Optional[str] = None,
    ):
        super().__init__(
            language,
            save_file,
            source_id,
            "broadcast news (and major networks)",
            overwrite,
            llm_id_search,
            llm_id_parse,
            log_dir,
        )


class LLMContentCreatorSource(BaseLLMSource):
    """LLM-based content creator source"""

    def __init__(
        self,
        language: str,
        save_file: str,
        source_id: str,
        overwrite: bool = False,
        llm_id_search: str = DEFAULT_SEARCH_LLM,
        llm_id_parse: str = DEFAULT_PARSE_LLM,
        log_dir: Optional[str] = None,
    ):
        super().__init__(
            language,
            save_file,
            source_id,
            "independent content creators (no major networks)",
            overwrite,
            llm_id_search,
            llm_id_parse,
            log_dir,
        )
