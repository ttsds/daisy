"""
Core data models for the Daisy pipeline.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional
import json
from importlib import resources
import numpy as np
import os
import soundfile
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


@dataclass
class Language(ABC):
    iso2: str
    iso3: str
    english_name: str
    native_name: str
    mms_code: Optional[str] = None


class FilterResult(BaseModel):
    identifier: str
    is_from_creator: bool
    is_right_language: bool
    mostly_spoken_content: bool
    category: str


class FilterResultList(BaseModel):
    items: list[FilterResult]


class MediaItem(BaseModel):
    """Unified data model for all media items"""

    identifier: str
    name: str
    description: str
    categories: list[str]
    language: str
    country: str


class DownloadItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sr: int
    audio: np.ndarray
    segment: Optional[tuple[int, int]] = None
    media_item_id: str
    audio_item_id: str
    identifier: str


class AudioItem(BaseModel):
    identifier: str
    title: str
    views: str
    date: str
    duration: str
    url: str
    channel_name: str
    url_id: str
    date_searched: Optional[datetime] = None
    parsed_views: Optional[int] = None
    parsed_duration: Optional[int] = None
    parsed_date: Optional[datetime] = None
    media_item_id: str


class FullItem(BaseModel):  # includes media item, audio item, and filter result
    media_item: MediaItem
    audio_item: AudioItem
    filter_result: FilterResult


class MediaItemList(BaseModel):
    """Generic list container for media items"""

    items: list[MediaItem]

class Utterance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    identifier: str
    audio: np.ndarray
    sr: int
    snr: float
    similarity: float
    duration: float
    texts: list[str]
    audio_item_id: str
    text: Optional[str] = None

class SimpleUtterance(BaseModel):
    identifier: str
    text: str

class UtteranceList(BaseModel):
    items: list[SimpleUtterance]

class Speaker(BaseModel):
    identifier: str
    name: str
    utterances: list[Utterance]

# Load language definitions
with resources.path("daisy.config", "languages.json") as path:
    with open(path, "r", encoding="utf-8") as f:
        LANGUAGES = {
            language["iso2"]: Language(**language) for language in json.load(f)
        }
