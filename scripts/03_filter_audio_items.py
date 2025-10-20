#!/usr/bin/env python3
"""
Stage 3: Filter Audio Items
===========================

This script filters audio items using LLM-based filtering to determine
which audio items are suitable for the dataset.

Usage:
    python scripts/03_filter_audio_items.py
"""

import os
import json
from multiprocessing import Pool, cpu_count

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from daisy.core import LANGUAGES, MediaItem, AudioItem
from daisy.processing import ResultsFilter
from dotenv import load_dotenv


def process_media_item(args):
    """
    Process a single media item for filtering.
    This function is designed to be used with multiprocessing.

    Args:
        args: Tuple containing (language, category, media_item_dict, media_item_index)

    Returns:
        Tuple containing (success, language, category, media_item_identifier, error_message)
    """
    _language, _category, media_item_dict, media_item_index = args

    try:
        # Create MediaItem from dict
        if "identifier" in media_item_dict:
            media_item = MediaItem(**media_item_dict)
        else:
            media_item = MediaItem(
                identifier=f"{_language}-{_category}-{media_item_index}",
                **media_item_dict,
            )

        # Check if filtered file already exists
        filtered_file_path = f"{DAISY_ROOT}/{_language}/{_category}-filtered/{media_item.identifier}.json"
        if os.path.exists(filtered_file_path):
            return (True, _language, _category, media_item.identifier, None)

        # Ensure filtered directory exists
        filtered_dir = f"{DAISY_ROOT}/{_language}/{_category}-filtered"
        os.makedirs(filtered_dir, exist_ok=True)

        # Load audio items
        audio_file_path = (
            f"{DAISY_ROOT}/{_language}/{_category}/{media_item.identifier}.json"
        )
        if not os.path.exists(audio_file_path):
            return (
                False,
                _language,
                _category,
                media_item.identifier,
                f"Audio file not found: {audio_file_path}",
            )

        with open(audio_file_path, "r", encoding="utf-8") as f:
            audio_items_data = json.load(f)

        audio_items = [AudioItem(**audio_item) for audio_item in audio_items_data]

        # Filter the audio items
        filter_results = ResultsFilter().filter(media_item, audio_items)

        # Save filtered results
        with open(filtered_file_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    filter_result.model_dump(mode="json")
                    for filter_result in filter_results.items
                ],
                f,
            )

        return (True, _language, _category, media_item.identifier, None)

    except Exception as e:
        return (
            False,
            _language,
            _category,
            media_item_dict.get("identifier", f"unknown-{media_item_index}"),
            str(e),
        )


def main():
    """Main function to filter audio items."""
    load_dotenv()

    # Validate required environment variables
    if not os.getenv("OPENROUTER_KEY"):
        raise ValueError("OPENROUTER_KEY environment variable is required")
    if not os.getenv("DAISY_ROOT"):
        raise ValueError("DAISY_ROOT environment variable is required")

    global DAISY_ROOT
    DAISY_ROOT = os.getenv("DAISY_ROOT")

    # Configuration
    NUM_CORES = (
        cpu_count() - 1
    )  # Use all cores except one to leave some for system processes
    # You can override this by setting NUM_CORES environment variable
    if os.getenv("NUM_CORES"):
        NUM_CORES = int(os.getenv("NUM_CORES"))

    print("Stage 3: Filtering Audio Items")
    print("=" * 40)
    print(f"Using {NUM_CORES} cores for parallel filtering")

    # Statistics tracking
    total_processed = 0
    total_errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        # Main language progress bar
        language_task = progress.add_task("Processing languages", total=len(LANGUAGES))

        for language_idx, language in enumerate(LANGUAGES):
            progress.update(
                language_task, description=f"Processing language: {language}"
            )

            # Category progress bar
            categories = ["podcasts", "broadcast_news", "content_creators"]
            category_task = progress.add_task(
                f"Processing categories for {language}", total=len(categories)
            )

            for category_idx, category in enumerate(categories):
                progress.update(
                    category_task, description=f"Processing {category} for {language}"
                )

                # Load media items for this category
                _media_items = json.load(
                    open(f"{DAISY_ROOT}/{language}/{category}.json", encoding="utf-8")
                )

                # Prepare arguments for parallel processing
                process_args = [
                    (language, category, media_item, i)
                    for i, media_item in enumerate(_media_items)
                ]

                # Media item progress bar
                media_task = progress.add_task(
                    f"Processing media items for {category}", total=len(process_args)
                )

                # Process media items in parallel
                print(f"Processing {len(process_args)} media items in parallel")
                with Pool(processes=NUM_CORES) as pool:
                    # Use imap to get results as they complete for progress tracking
                    results = pool.imap(process_media_item, process_args)

                    for result in results:
                        success, result_language, result_category, identifier, error = (
                            result
                        )
                        total_processed += 1

                        if success:
                            progress.update(
                                media_task, description=f"Completed {identifier}"
                            )
                        else:
                            total_errors += 1
                            progress.update(
                                media_task, description=f"Failed {identifier}: {error}"
                            )
                            print(f"Error processing {identifier}: {error}")

                        progress.advance(media_task)

                progress.remove_task(media_task)
                progress.advance(category_task)

            progress.remove_task(category_task)
            progress.advance(language_task)

    # Print final statistics
    print("\nStage 3 completed! Audio items filtered for all languages.")
    print(f"Total items processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(
        f"Success rate: {((total_processed - total_errors) / total_processed * 100):.1f}%"
        if total_processed > 0
        else "No items processed"
    )
    print("Next: Run 'python scripts/04_sample_audio_items.py' to sample audio items.")


if __name__ == "__main__":
    main()
