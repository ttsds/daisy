#!/usr/bin/env python3
"""
Stage 4: Sample Audio Items
===========================

This script samples audio items from the dataset to create a smaller dataset.

Usage:
    python scripts/04_sample_audio_items.py
"""

import os
import json
from glob import glob

import rich.progress

from daisy.processing import ResultSampler
from daisy.core import LANGUAGES, MediaItem, AudioItem, FilterResult
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

    return podcast_items + news_items + creator_items


def load_audio_items(language: str):
    podcast_items = []
    news_items = []
    creator_items = []
    for file in glob(f"{DAISY_ROOT}/{language}/podcasts/*.json"):
        with open(file, "r", encoding="utf-8") as f:
            podcast_items.extend(json.load(f))
    for file in glob(f"{DAISY_ROOT}/{language}/broadcast_news/*.json"):
        with open(file, "r", encoding="utf-8") as f:
            news_items.extend(json.load(f))
    for file in glob(f"{DAISY_ROOT}/{language}/content_creators/*.json"):
        with open(file, "r", encoding="utf-8") as f:
            creator_items.extend(json.load(f))

    podcast_items_new = []
    for item in podcast_items:
        podcast_items_new.append(AudioItem(**item))
    news_items_new = []
    for item in news_items:
        news_items_new.append(AudioItem(**item))
    creator_items_new = []
    for item in creator_items:
        creator_items_new.append(AudioItem(**item))
    return podcast_items_new + news_items_new + creator_items_new


def load_filter_results(language: str):
    podcast_filter_results = []
    news_filter_results = []
    creator_filter_results = []
    for file in glob(f"{DAISY_ROOT}/{language}/podcasts-filtered/*.json"):
        with open(file, "r", encoding="utf-8") as f:
            podcast_filter_results.extend(json.load(f))
    for file in glob(f"{DAISY_ROOT}/{language}/broadcast_news-filtered/*.json"):
        with open(file, "r", encoding="utf-8") as f:
            news_filter_results.extend(json.load(f))
    for file in glob(f"{DAISY_ROOT}/{language}/content_creators-filtered/*.json"):
        with open(file, "r", encoding="utf-8") as f:
            creator_filter_results.extend(json.load(f))

    podcast_filter_results_new = []
    news_filter_results_new = []
    creator_filter_results_new = []
    for item in podcast_filter_results:
        podcast_filter_results_new.append(FilterResult(**item))
    for item in news_filter_results:
        news_filter_results_new.append(FilterResult(**item))
    for item in creator_filter_results:
        creator_filter_results_new.append(FilterResult(**item))
    return (
        podcast_filter_results_new
        + news_filter_results_new
        + creator_filter_results_new
    )


def main():
    """Main function to sample audio items."""
    load_dotenv()

    # Validate required environment variables
    if not os.getenv("DAISY_ROOT"):
        raise ValueError("DAISY_ROOT environment variable is required")

    global DAISY_ROOT
    DAISY_ROOT = os.getenv("DAISY_ROOT")

    print("Stage 4: Sampling Audio Items")
    print("=" * 40)

    for language in rich.progress.track(LANGUAGES):
        print(f"\nProcessing language: {language}")
        # if os.path.exists(f"{DAISY_ROOT}/{language}/sampled_items.csv"):
        #     continue
        media_items = load_media_items(language)
        audio_items = load_audio_items(language)
        filter_results = load_filter_results(language)
        result_sampler = ResultSampler(media_items, audio_items, filter_results)
        sampled_items, sampled_items_df = result_sampler.sample(100, return_df=True)
        sampled_items_df.to_csv(
            f"{DAISY_ROOT}/{language}/sampled_items.csv", index=False
        )

    print("\nStage 4 completed! Audio items sampled for all languages.")
    print("Next: Run 'python scripts/05_download_samples.py' to download samples.")


if __name__ == "__main__":
    main()
