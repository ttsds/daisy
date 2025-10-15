#!/usr/bin/env python3
"""
Stage 1: Collect Media Sources
=============================

This script collects media sources (podcasts, broadcast news, content creators)
for each language using LLM-based sources.

Usage:
    python scripts/01_collect_media_sources.py
"""

import os
import rich
from daisy.media_sources import (
    LLMPodcastSource,
    LLMBroadcastNewsSource,
    LLMContentCreatorSource,
)
from daisy.abstract import LANGUAGES
from dotenv import load_dotenv


def main():
    """Main function to collect media sources."""
    load_dotenv()

    # Validate required environment variables
    if not os.getenv("OPENROUTER_KEY"):
        raise ValueError("OPENROUTER_KEY environment variable is required")
    if not os.getenv("DAISY_ROOT"):
        raise ValueError("DAISY_ROOT environment variable is required")

    DAISY_ROOT = os.getenv("DAISY_ROOT")

    print("Stage 1: Collecting Media Sources")
    print("=" * 40)

    for language in rich.progress.track(
        LANGUAGES, description="Collecting media sources"
    ):
        print(f"\nProcessing language: {language}")

        # Collect podcasts
        podcasts = LLMPodcastSource(
            language=language,
            save_file=f"{DAISY_ROOT}/{language}/podcasts.json",
            source_id="podcasts",
            log_dir=f"{DAISY_ROOT}/logs/{language}",
        )
        podcast_items = podcasts.collect()
        print(f"Collected {len(podcast_items)} podcasts for {language}")

        # Collect broadcast news
        broadcast_news = LLMBroadcastNewsSource(
            language=language,
            save_file=f"{DAISY_ROOT}/{language}/broadcast_news.json",
            source_id="broadcast_news",
            log_dir=f"{DAISY_ROOT}/logs/{language}",
        )
        news_items = broadcast_news.collect()
        print(f"Collected {len(news_items)} broadcast news for {language}")

        # Collect content creators
        content_creators = LLMContentCreatorSource(
            language=language,
            save_file=f"{DAISY_ROOT}/{language}/content_creators.json",
            source_id="content_creators",
            log_dir=f"{DAISY_ROOT}/logs/{language}",
        )
        creator_items = content_creators.collect()
        print(f"Collected {len(creator_items)} content creators for {language}")

    print("\nStage 1 completed! Media sources collected for all languages.")
    print(
        "Next: Run 'python scripts/02_collect_audio_items.py' to collect audio items."
    )


if __name__ == "__main__":
    main()
