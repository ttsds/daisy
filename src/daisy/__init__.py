"""
Daisy: A diverse and multilingual speech dataset for evaluation of speech models.

This package provides tools for collecting, filtering, and downloading multilingual
speech data from various sources including podcasts, broadcast news, and content creators.
"""

from .core import (
    Language,
    MediaItem,
    AudioItem,
    FilterResult,
    DownloadItem,
    LANGUAGES,
)
from .sources.media import (
    LLMPodcastSource,
    LLMBroadcastNewsSource,
    LLMContentCreatorSource,
)
from .sources.audio import (
    YouTubeAudioSource,
    BilibiliAudioSource,
)
from .download import (
    VideoAudioDownloader,
)
from .processing import (
    ResultsFilter,
    ResultSampler,
    NeuralDiarizer,
    ClusteringDiarizer,
)
from .processing import SpeakerExtractor
from .utils import (
    find_valleys,
    wada_snr,
    DemucsProcessor,
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
    # Diarization
    "NeuralDiarizer",
    "ClusteringDiarizer",
    # Utterance pipeline
    "SpeakerExtractor",
    # Audio processing utilities
    "find_valleys",
    "wada_snr",
    "DemucsProcessor",
]