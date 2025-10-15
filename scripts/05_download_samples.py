#!/usr/bin/env python3
"""
Stage 5: Download Samples
=========================

This script downloads the sampled audio items.

Usage:
    python scripts/05_download_samples.py
"""

import os
import pandas as pd

from daisy.downloaders import VideoAudioDownloader
from daisy.abstract import AudioItem, LANGUAGES
from dotenv import load_dotenv


def load_audio_items(language: str):
    """Load audio items from the sampled CSV file."""
    audio_items = pd.read_csv(f"{DAISY_ROOT}/{language}/sampled_items.csv")
    audio_items_new = []
    for _, row in audio_items.iterrows():
        row["identifier"] = row["audio_item_id"]
        row["views"] = str(row["views"])
        row["duration"] = str(row["duration"])
        if "youtube.com" in row["url"]:
            row["url_id"] = row["url"].split("v=")[-1]
        elif "bilibili.com" in row["url"]:
            row["url_id"] = row["url"].split("/")[-2]
        audio_items_new.append(AudioItem(**row.to_dict()))
    return audio_items_new


def main():
    """Main function to download samples."""
    load_dotenv()

    # Validate required environment variables
    if not os.getenv("DAISY_ROOT"):
        raise ValueError("DAISY_ROOT environment variable is required")

    global DAISY_ROOT
    DAISY_ROOT = os.getenv("DAISY_ROOT")

    print("Stage 5: Downloading Samples")
    print("=" * 40)

    for language in LANGUAGES:
        print(f"\n=== Downloading samples for {language} ===")
        downloader = VideoAudioDownloader(save_dir=f"{DAISY_ROOT}/{language}/samples")
        audio_items = load_audio_items(language)
        print(f"Found {len(audio_items)} audio items to download")
        downloader.collect(audio_items, use_multiprocessing=True, show_progress=True)

    print("\nStage 5 completed! Samples downloaded for all languages.")
    print("Dataset creation pipeline completed!")


if __name__ == "__main__":
    main()
