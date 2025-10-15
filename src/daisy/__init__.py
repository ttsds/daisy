"""
Daisy: A diverse and multilingual speech dataset for evaluation of speech models.

This package provides tools for collecting, filtering, and downloading multilingual
speech data from various sources including podcasts, broadcast news, and content creators.
"""

from .abstract import (
    Language,
    MediaItem,
    AudioItem,
    FilterResult,
    DownloadItem,
    LANGUAGES,
)
from .media_sources import (
    LLMPodcastSource,
    LLMBroadcastNewsSource,
    LLMContentCreatorSource,
)
from .audio_sources import (
    YouTubeAudioSource,
    BilibiliAudioSource,
)
from .downloaders import (
    VideoAudioDownloader,
)
from .filter_results import (
    ResultsFilter,
)
from .sample_results import (
    ResultSampler,
)

__version__ = "0.0.1"
__author__ = "Christoph Minixhofer"
__email__ = "christoph.minixhofer@gmail.com"

__all__ = [
    # Data models
    "Language",
    "MediaItem",
    "AudioItem",
    "FilterResult",
    "DownloadItem",
    "LANGUAGES",
    # Media sources
    "LLMPodcastSource",
    "LLMBroadcastNewsSource",
    "LLMContentCreatorSource",
    # Audio sources
    "YouTubeAudioSource",
    "BilibiliAudioSource",
    # Downloaders
    "VideoAudioDownloader",
    # Filters
    "ResultsFilter",
    # Sampling
    "ResultSampler",
]
