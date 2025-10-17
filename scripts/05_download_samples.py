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
import threading
import time
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

from daisy.downloaders import VideoAudioDownloader
from daisy.abstract import AudioItem, LANGUAGES
from dotenv import load_dotenv


class DownloadMonitor:
    """Monitor thread that tracks download progress for each language."""

    def __init__(self, daisy_root: str, languages: list[str], log_file: str):
        self.daisy_root = daisy_root
        self.languages = languages
        self.log_file = log_file
        self.running = False
        self.monitor_thread = None
        self.total_counts = {}  # Store total expected counts for each language

        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def count_audio_files(self, language: str) -> int:
        """Count the number of .wav files in the samples directory for a language."""
        samples_dir = Path(self.daisy_root) / language / "samples"
        if not samples_dir.exists():
            return 0
        return len(list(samples_dir.glob("*.wav")))

    def load_total_counts(self):
        """Load the total expected counts for each language from sampled_items.csv files."""
        for language in self.languages:
            csv_file = Path(self.daisy_root) / language / "sampled_items.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    self.total_counts[language] = len(df)
                except Exception as e:
                    print(f"Warning: Could not read {csv_file}: {e}")
                    self.total_counts[language] = 0
            else:
                self.total_counts[language] = 0

    def write_progress_log(self, progress_data: dict):
        """Write progress data to the log file in table format (overwrites file)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare table data
        table_data = []
        for language in self.languages:
            downloaded = progress_data.get(language, 0)
            total = self.total_counts.get(language, 0)
            percentage = (downloaded / total * 100) if total > 0 else 0

            table_data.append(
                [
                    language.upper(),
                    f"{downloaded:,}",
                    f"{total:,}",
                    f"{percentage:.1f}%",
                ]
            )

        # Create table
        headers = ["Language", "Downloaded", "Total", "Progress"]
        table = tabulate(table_data, headers=headers, tablefmt="grid", stralign="right")

        # Write to file (overwrite)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Download Progress Report - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(table + "\n\n")
            f.write(f"Last updated: {timestamp}\n")

    def monitor_loop(self):
        """Main monitoring loop that runs every second."""
        while self.running:
            progress_data = {}
            for language in self.languages:
                count = self.count_audio_files(language)
                progress_data[language] = count

            self.write_progress_log(progress_data)

            # Clear screen and display table in console
            os.system("clear" if os.name == "posix" else "cls")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Download Progress - {timestamp}")
            print("=" * 50)

            # Prepare table data for console
            table_data = []
            for language in self.languages:
                downloaded = progress_data.get(language, 0)
                total = self.total_counts.get(language, 0)
                percentage = (downloaded / total * 100) if total > 0 else 0

                table_data.append(
                    [
                        language.upper(),
                        f"{downloaded:,}",
                        f"{total:,}",
                        f"{percentage:.1f}%",
                    ]
                )

            # Display table
            headers = ["Language", "Downloaded", "Total", "Progress"]
            table = tabulate(
                table_data, headers=headers, tablefmt="grid", stralign="right"
            )

            time.sleep(1)

    def start(self):
        """Start the monitoring thread."""
        # Load total counts first
        self.load_total_counts()

        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Started download monitor. Logging to: {self.log_file}")

    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("Download monitor stopped.")


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

    # Initialize and start the download monitor
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/download_progress.log"
    monitor = DownloadMonitor(DAISY_ROOT, list(LANGUAGES.keys()), log_file)
    monitor.start()

    try:
        for language in LANGUAGES:
            print(f"\n=== Downloading samples for {language} ===")
            downloader = VideoAudioDownloader(
                save_dir=f"{DAISY_ROOT}/{language}/samples"
            )
            audio_items = load_audio_items(language)
            print(f"Found {len(audio_items)} audio items to download")
            downloader.collect(
                audio_items, use_multiprocessing=True, show_progress=True
            )

        print("\nStage 5 completed! Samples downloaded for all languages.")
        print("Dataset creation pipeline completed!")

    finally:
        # Always stop the monitor when done
        monitor.stop()


if __name__ == "__main__":
    main()
