# Daisy API Reference

## Core Classes

### Data Models

#### `Language`
A dataclass representing language information.

```python
from daisy import Language

# Access predefined languages
from daisy import LANGUAGES
arabic = LANGUAGES["ar"]
print(arabic.english_name)  # "Arabic"
print(arabic.native_name)  # "العربية"
```

#### `MediaItem`
Represents a media source (podcast, broadcast news, content creator).

```python
from daisy import MediaItem

item = MediaItem(
    identifier="test-1",
    name="Test Podcast",
    description="A test podcast",
    categories=["news"],
    language="en",
    country="US"
)
```

#### `AudioItem`
Represents an audio item found from a media source.

```python
from daisy import AudioItem

item = AudioItem(
    identifier="audio-1",
    title="Test Audio",
    views="1000",
    date="2025-01-01",
    duration="120",
    url="https://example.com/abc123",
    channel_name="Test Channel",
    url_id="abc123",
    media_item_id="media-1"
)
```

### Media Sources

#### `LLMPodcastSource`
Collects podcast sources using LLM-based search.

```python
from daisy import LLMPodcastSource

source = LLMPodcastSource(
    language="en",
    save_file="/path/to/podcasts.json",
    source_id="podcasts"
)
items = source.collect()
```

#### `LLMBroadcastNewsSource`
Collects broadcast news sources using LLM-based search.

```python
from daisy import LLMBroadcastNewsSource

source = LLMBroadcastNewsSource(
    language="en",
    save_file="/path/to/news.json",
    source_id="news"
)
items = source.collect()
```

#### `LLMContentCreatorSource`
Collects content creator sources using LLM-based search.

```python
from daisy import LLMContentCreatorSource

source = LLMContentCreatorSource(
    language="en",
    save_file="/path/to/creators.json",
    source_id="creators"
)
items = source.collect()
```

### Audio Sources

#### `YouTubeAudioSource`
Searches for audio items on YouTube.

```python
from daisy import YouTubeAudioSource

source = YouTubeAudioSource(
    save_file="/path/to/audio.json",
    language="en",
    date=(start_date, end_date)
)
items = source.collect(media_item)
```

#### `BilibiliAudioSource`
Searches for audio items on Bilibili.

```python
from daisy import BilibiliAudioSource

source = BilibiliAudioSource(
    save_file="/path/to/audio.json",
    language="cmn-cn",
    date=(start_date, end_date)
)
items = source.collect(media_item)
```

### Downloaders

#### `VideoAudioDownloader`
Downloads audio from video URLs.

```python
from daisy import VideoAudioDownloader

downloader = VideoAudioDownloader(
    save_dir="/path/to/samples",
    overwrite=False
)
downloader.collect(audio_items)
```

### Filters

#### `ResultsFilter`
Filters audio items using LLM-based criteria.

```python
from daisy import ResultsFilter

filter_source = ResultsFilter(
    language="en",
    save_file="/path/to/filtered.json",
    source_id="filter"
)
filtered_items = filter_source.collect(audio_items)
```

### Sampling

#### `ResultSampler`
Samples audio items to create a smaller dataset.

```python
from daisy import ResultSampler

sampler = ResultSampler(
    language="en",
    save_file="/path/to/sampled.csv"
)
sampled_items = sampler.collect(filtered_items)
```

## Environment Variables

- `OPENROUTER_KEY`: Required API key for OpenRouter
- `DAISY_ROOT`: Root directory for dataset storage

## Constants

- `LANGUAGES`: Dictionary of supported languages (69 languages total)
- See the available languages in `src/daisy/data/languages.json`
