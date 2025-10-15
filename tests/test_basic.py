"""
Basic tests for Daisy package.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from daisy.abstract import LANGUAGES, MediaItem, AudioItem, Language
from daisy.media_sources import (
    LLMPodcastSource,
    LLMBroadcastNewsSource,
    LLMContentCreatorSource,
)


class TestAbstract:
    """Test abstract classes and data models."""

    def test_languages_loaded(self):
        """Test that languages are properly loaded."""
        assert len(LANGUAGES) > 0
        assert "en" in LANGUAGES
        assert "es" in LANGUAGES

    def test_language_structure(self):
        """Test Language dataclass structure."""
        english = LANGUAGES["en"]
        assert hasattr(english, "iso2")
        assert hasattr(english, "iso3")
        assert hasattr(english, "english_name")
        assert hasattr(english, "native_name")
        assert english.iso2 == "en"
        assert english.english_name == "English"

    def test_media_item_creation(self):
        """Test MediaItem creation."""
        item = MediaItem(
            identifier="test-1",
            name="Test Podcast",
            description="A test podcast",
            categories=["news"],
            language="en",
            country="US",
        )
        assert item.identifier == "test-1"
        assert item.name == "Test Podcast"
        assert item.language == "en"

    def test_audio_item_creation(self):
        """Test AudioItem creation."""
        item = AudioItem(
            identifier="audio-1",
            title="Test Audio",
            views="1000",
            date="2025-01-01",
            duration="120",
            url="https://example.com",
            channel_name="Test Channel",
            url_id="abc123",
            media_item_id="media-1",
        )
        assert item.identifier == "audio-1"
        assert item.title == "Test Audio"
        assert item.views == "1000"


class TestMediaSources:
    """Test media source classes."""

    @patch.dict(os.environ, {"OPENROUTER_KEY": "test-key", "DAISY_ROOT": "/tmp/test"})
    def test_llm_podcast_source_init(self):
        """Test LLMPodcastSource initialization."""
        source = LLMPodcastSource(
            language="en", save_file="/tmp/test.json", source_id="test"
        )
        assert source.language.iso2 == "en"
        assert source.content_type == "podcasts"

    @patch.dict(os.environ, {"OPENROUTER_KEY": "test-key", "DAISY_ROOT": "/tmp/test"})
    def test_llm_broadcast_news_source_init(self):
        """Test LLMBroadcastNewsSource initialization."""
        source = LLMBroadcastNewsSource(
            language="en", save_file="/tmp/test.json", source_id="test"
        )
        assert source.language.iso2 == "en"
        assert source.content_type == "broadcast news (and major networks)"

    @patch.dict(os.environ, {"OPENROUTER_KEY": "test-key", "DAISY_ROOT": "/tmp/test"})
    def test_llm_content_creator_source_init(self):
        """Test LLMContentCreatorSource initialization."""
        source = LLMContentCreatorSource(
            language="en", save_file="/tmp/test.json", source_id="test"
        )
        assert source.language.iso2 == "en"
        assert source.content_type == "independent content creators (no major networks)"


class TestEnvironmentValidation:
    """Test environment variable validation."""

    def test_missing_openrouter_key(self):
        """Test that missing OPENROUTER_KEY raises error during OpenAI client creation."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):  # OpenAIError or similar
                LLMPodcastSource(
                    language="en", save_file="/tmp/test.json", source_id="test"
                )

    def test_llm_source_initialization_with_keys(self):
        """Test that LLM sources initialize correctly with proper environment variables."""
        with patch.dict(os.environ, {"OPENROUTER_KEY": "test-key"}, clear=True):
            source = LLMPodcastSource(
                language="en", save_file="/tmp/test.json", source_id="test"
            )
            assert source.language.iso2 == "en"


if __name__ == "__main__":
    pytest.main([__file__])
