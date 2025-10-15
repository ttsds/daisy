#!/usr/bin/env python3
"""
Stage 2: Collect Audio Items
============================

This script collects audio items from YouTube/Bilibili for each media source
that was collected in Stage 1.

Usage:
    python scripts/02_collect_audio_items.py
"""

import os
import json
from datetime import datetime, timedelta

import rich.progress

from daisy.audio_sources import YouTubeAudioSource, BilibiliAudioSource
from daisy.abstract import LANGUAGES, MediaItem
from dotenv import load_dotenv


def load_media_items(language: str):
    """Load media items from the JSON files created in Stage 1."""
    with open(f"{DAISY_ROOT}/{language}/podcasts.json", "r", encoding="utf-8") as f:
        podcast_items = json.load(f)
        podcast_items_media = []
        for i, podcast in enumerate(podcast_items):
            if "identifier" in podcast:
                podcast_items_media.append(MediaItem(**podcast))
            else:
                podcast_items_media.append(
                    MediaItem(identifier=f"{language}-podcasts-{i}", **podcast)
                )
        podcast_items = podcast_items_media

    with open(
        f"{DAISY_ROOT}/{language}/broadcast_news.json", "r", encoding="utf-8"
    ) as f:
        news_items = json.load(f)
        news_items_media = []
        for i, news in enumerate(news_items):
            if "identifier" in news:
                news_items_media.append(MediaItem(**news))
            else:
                news_items_media.append(
                    MediaItem(identifier=f"{language}-broadcast_news-{i}", **news)
                )
        news_items = news_items_media

    with open(
        f"{DAISY_ROOT}/{language}/content_creators.json", "r", encoding="utf-8"
    ) as f:
        creator_items = json.load(f)
        creator_items_media = []
        for i, creator in enumerate(creator_items):
            if "identifier" in creator:
                creator_items_media.append(MediaItem(**creator))
            else:
                creator_items_media.append(
                    MediaItem(identifier=f"{language}-content_creators-{i}", **creator)
                )
        creator_items = creator_items_media

    return podcast_items, news_items, creator_items


def main():
    """Main function to collect audio items."""
    load_dotenv()

    # Validate required environment variables
    if not os.getenv("DAISY_ROOT"):
        raise ValueError("DAISY_ROOT environment variable is required")

    global DAISY_ROOT
    DAISY_ROOT = os.getenv("DAISY_ROOT")

    # Configuration
    bilibili_langs = ["cmn-cn", "yue-cn"]
    after_date = datetime.now() - timedelta(days=60)

    print("Stage 2: Collecting Audio Items")
    print("=" * 40)

    for language in rich.progress.track(
        LANGUAGES, description="Collecting audio items"
    ):
        print(f"\nProcessing language: {language}")

        podcast_items, news_items, creator_items = load_media_items(language)

        if language in bilibili_langs:
            print(f"Using Bilibili for {language}")

            # Collect podcasts from Bilibili
            os.makedirs(f"{DAISY_ROOT}/{language}/podcasts", exist_ok=True)
            for podcast in podcast_items:
                bilibili_items = BilibiliAudioSource(
                    save_file=f"{DAISY_ROOT}/{language}/podcasts/{podcast.identifier}.json",
                    language=language,
                    date=(after_date, datetime.now()),
                ).collect(podcast)
                print(
                    f"Collected {len(bilibili_items)} podcasts bilibili audio for {language}"
                )

            # Collect broadcast news from Bilibili
            os.makedirs(f"{DAISY_ROOT}/{language}/broadcast_news", exist_ok=True)
            for news in news_items:
                bilibili_items = BilibiliAudioSource(
                    save_file=f"{DAISY_ROOT}/{language}/broadcast_news/{news.identifier}.json",
                    language=language,
                    date=(after_date, datetime.now()),
                ).collect(news)
                print(
                    f"Collected {len(bilibili_items)} broadcast news bilibili audio for {language}"
                )

            # Collect content creators from Bilibili
            os.makedirs(f"{DAISY_ROOT}/{language}/content_creators", exist_ok=True)
            for creator in creator_items:
                bilibili_items = BilibiliAudioSource(
                    save_file=f"{DAISY_ROOT}/{language}/content_creators/{creator.identifier}.json",
                    language=language,
                    date=(after_date, datetime.now()),
                ).collect(creator)
                print(
                    f"Collected {len(bilibili_items)} content creators bilibili audio for {language}"
                )
        else:
            print(f"Using YouTube for {language}")

            # Collect podcasts from YouTube
            os.makedirs(f"{DAISY_ROOT}/{language}/podcasts", exist_ok=True)
            for podcast in podcast_items:
                youtube_items = YouTubeAudioSource(
                    save_file=f"{DAISY_ROOT}/{language}/podcasts/{podcast.identifier}.json",
                    language=language,
                    date=(after_date, None),
                ).collect(podcast)
                print(f"Collected {len(youtube_items)} youtube audio for {language}")

            # Collect broadcast news from YouTube
            os.makedirs(f"{DAISY_ROOT}/{language}/broadcast_news", exist_ok=True)
            for news in news_items:
                youtube_items = YouTubeAudioSource(
                    save_file=f"{DAISY_ROOT}/{language}/broadcast_news/{news.identifier}.json",
                    language=language,
                    date=(after_date, None),
                ).collect(news)
                print(f"Collected {len(youtube_items)} youtube audio for {language}")

            # Collect content creators from YouTube
            os.makedirs(f"{DAISY_ROOT}/{language}/content_creators", exist_ok=True)
            for creator in creator_items:
                youtube_items = YouTubeAudioSource(
                    save_file=f"{DAISY_ROOT}/{language}/content_creators/{creator.identifier}.json",
                    language=language,
                    date=(after_date, None),
                ).collect(creator)
                print(f"Collected {len(youtube_items)} youtube audio for {language}")

    print("\nStage 2 completed! Audio items collected for all languages.")
    print("Next: Run 'python scripts/03_filter_audio_items.py' to filter audio items.")


if __name__ == "__main__":
    main()
