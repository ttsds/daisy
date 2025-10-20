"""
Audio source collection modules.
"""

from .youtube import YouTubeAudioSource
from .bilibili import BilibiliAudioSource

__all__ = [
    "YouTubeAudioSource",
    "BilibiliAudioSource",
]
