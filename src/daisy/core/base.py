"""
Abstract base classes for the Daisy pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import os
import json
import soundfile
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from .models import MediaItem, AudioItem, DownloadItem, LANGUAGES


def _download_single_item(
    downloader_instance, item: AudioItem
) -> Optional[DownloadItem]:
    """
    Helper function for multiprocessing downloads.
    This function needs to be at module level for pickle serialization.
    """
    try:
        if (
            os.path.exists(
                os.path.join(downloader_instance.save_dir, f"{item.identifier}.wav")
            )
            and not downloader_instance.overwrite
        ):
            return "skipped"  # Return special value to indicate skipped file

        if any(host in item.url for host in downloader_instance.hosts_supported):
            download_item = downloader_instance.download(item)
            if download_item is None:
                return "failed"  # Return special value to indicate failed download
        else:
            raise ValueError(f"Host {item.url} not supported")

        download_item.media_item_id = item.media_item_id
        download_item.audio_item_id = item.identifier

        # Save the audio file
        os.makedirs(downloader_instance.save_dir, exist_ok=True)
        save_file = os.path.join(
            downloader_instance.save_dir, f"{download_item.identifier}.wav"
        )
        if os.path.exists(save_file) and not downloader_instance.overwrite:
            return "skipped"

        soundfile.write(save_file, download_item.audio, download_item.sr)
        return download_item

    except Exception as e:
        print(f"Error downloading {item.identifier}: {e}")
        return "failed"


class AudioSource(ABC):
    def __init__(
        self,
        save_file: str,
        language: str,
        overwrite: bool = False,
        **filters: Any,
    ):
        """
        Initialize the web source and filters.
        Filters are used to filter the results of the search, e.g. number of youtube views, length of the audio, etc.
        - "save_dir": str
        - "overwrite": bool
        - "creator": Optional[str] - Name of creator to filter by (e.g., "BBC", "NPR", "Joe Rogan")
        - "llm_id_filter": str - LLM model ID for creator filtering
        - "filters": Any
        """
        self.filters = filters
        self.save_file = save_file
        self.overwrite = overwrite
        self.language = LANGUAGES[language]

    @abstractmethod
    def search(self, item: MediaItem) -> list[AudioItem]:
        """
        Search for the query in the web source.
        Return a list of urls.
        - "item": MediaItem
        - "items": list[AudioItem]
        """
        pass

    @abstractmethod
    def filter(self, items: list[AudioItem]) -> list[AudioItem]:
        """
        Filter the urls based on the filters.
        Return a list of items.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the audio source.
        """
        pass

    @abstractmethod
    def init_driver(self) -> None:
        """
        Initialize the driver.
        """
        pass

    def collect(self, item: MediaItem) -> list[AudioItem]:
        if os.path.exists(self.save_file) and not self.overwrite:
            with open(self.save_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            items = [AudioItem(**item) for item in items]
            for i, _item in enumerate(items):
                _item.identifier = f"{_item.media_item_id}-{i}"
            # if len(items) > 0:
            self.close()
            return items
        self.init_driver()
        items = self.search(item)
        items = self.filter(items)
        with open(self.save_file, "w", encoding="utf-8") as f:
            json.dump([item.model_dump(mode="json") for item in items], f)
        for i, _item in enumerate(items):
            _item.identifier = f"{_item.media_item_id}-{i}"

        self.close()
        return items


class AudioDownloader(ABC):
    def __init__(
        self,
        save_dir: str,
        overwrite: bool = False,
        hosts_supported: list[str] = [],
        max_workers: Optional[int] = None,
    ):
        self.save_dir = save_dir
        self.overwrite = overwrite
        self.hosts_supported = hosts_supported
        self.max_workers = max_workers or min(
            cpu_count(), 4
        )  # Default to min of CPU count and 4

    @abstractmethod
    def download(self, item: AudioItem) -> DownloadItem:
        pass

    def collect(
        self,
        items: list[AudioItem],
        use_multiprocessing: bool = True,
        show_progress: bool = True,
    ) -> None:
        """
        Download audio items, optionally using multiprocessing for better performance.

        Args:
            items: List of AudioItem objects to download
            use_multiprocessing: Whether to use multiprocessing (default: True)
            show_progress: Whether to show progress bar (default: True)
        """
        if not items:
            return

        if use_multiprocessing and len(items) > 1:
            self._collect_multiprocessing(items, show_progress)
        else:
            self._collect_sequential(items, show_progress)

    def _collect_sequential(
        self, items: list[AudioItem], show_progress: bool = True
    ) -> None:
        """Sequential download (original implementation)"""
        if show_progress:
            progress_bar = tqdm(items, desc="Downloading", unit="item")
        else:
            progress_bar = items

        successful_downloads = 0
        skipped_downloads = 0

        for item in progress_bar:
            if (
                os.path.exists(os.path.join(self.save_dir, f"{item.identifier}.wav"))
                and not self.overwrite
            ):
                skipped_downloads += 1
                if show_progress:
                    progress_bar.set_postfix(
                        {
                            "successful": successful_downloads,
                            "skipped": skipped_downloads,
                            "current": (
                                item.title[:30] + "..."
                                if len(item.title) > 30
                                else item.title
                            ),
                        }
                    )
                continue

            os.makedirs(self.save_dir, exist_ok=True)
            if any(host in item.url for host in self.hosts_supported):
                download_item = self.download(item)
                if download_item is None:
                    if show_progress:
                        progress_bar.set_postfix(
                            {
                                "successful": successful_downloads,
                                "skipped": skipped_downloads,
                                "current": f"Failed: {item.title[:20]}...",
                            }
                        )
                    continue
            else:
                raise ValueError(f"Host {item.url} not supported")
            download_item.media_item_id = item.media_item_id
            download_item.audio_item_id = item.identifier
            save_file = os.path.join(self.save_dir, f"{download_item.identifier}.wav")
            if os.path.exists(save_file) and not self.overwrite:
                skipped_downloads += 1
                if show_progress:
                    progress_bar.set_postfix(
                        {
                            "successful": successful_downloads,
                            "skipped": skipped_downloads,
                            "current": (
                                item.title[:30] + "..."
                                if len(item.title) > 30
                                else item.title
                            ),
                        }
                    )
                continue
            soundfile.write(save_file, download_item.audio, download_item.sr)
            successful_downloads += 1

            if show_progress:
                progress_bar.set_postfix(
                    {
                        "successful": successful_downloads,
                        "skipped": skipped_downloads,
                        "current": (
                            item.title[:30] + "..."
                            if len(item.title) > 30
                            else item.title
                        ),
                    }
                )

        if show_progress:
            print(
                f"\nSequential download completed: {successful_downloads} successful, {skipped_downloads} skipped"
            )

    def _collect_multiprocessing(
        self, items: list[AudioItem], show_progress: bool = True
    ) -> None:
        """Multiprocessing download implementation"""
        if show_progress:
            print(
                f"Starting multiprocessing download with {self.max_workers} workers for {len(items)} items"
            )

        # Create a partial function with the downloader instance bound
        download_func = partial(_download_single_item, self)

        # Use multiprocessing Pool with progress tracking
        if show_progress:
            with Pool(processes=self.max_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(download_func, items),
                        total=len(items),
                        desc="Downloading",
                        unit="item",
                    )
                )
        else:
            with Pool(processes=self.max_workers) as pool:
                results = pool.map(download_func, items)

        # Count successful downloads, failures, and skipped items
        successful_downloads = sum(
            1 for result in results if isinstance(result, DownloadItem)
        )
        failed_downloads = sum(1 for result in results if result == "failed")
        skipped_downloads = sum(1 for result in results if result == "skipped")

        if show_progress:
            print(
                f"\nMultiprocessing download completed: {successful_downloads} successful, {failed_downloads} failed, {skipped_downloads} skipped"
            )
        else:
            print(
                f"Download results: {successful_downloads} successful, {failed_downloads} failed, {skipped_downloads} skipped"
            )


class ListSource(ABC):
    def __init__(
        self, language: str, save_file: str, source_id: str, overwrite: bool = False
    ):
        self.language = LANGUAGES[language]
        self.save_file = save_file
        self.source_id = source_id
        self.overwrite = overwrite

    @abstractmethod
    def search(self) -> list[MediaItem]:
        pass

    def collect(self) -> list[MediaItem]:
        if os.path.exists(self.save_file) and not self.overwrite:
            # Load existing items from file
            with open(self.save_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert to MediaItem objects
            media_items = []
            for i, item_data in enumerate(data):
                item_data["identifier"] = f"{self.language.iso2}-{self.source_id}-{i}"
                media_item = MediaItem(**item_data)
                media_items.append(media_item)
            return media_items

        # Get new items from search
        items = self.search()

        # Ensure the directory exists before writing the file
        save_dir = os.path.dirname(self.save_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Process and save items
        saved_items = []
        for i, item in enumerate(items):
            # Set identifier
            item_dict = item.model_dump(mode="json")
            item_dict["identifier"] = f"{self.language.iso2}-{self.source_id}-{i}"

            # Convert datetime objects to ISO strings for JSON serialization
            for key, value in item_dict.items():
                if isinstance(value, datetime):
                    item_dict[key] = value.isoformat()

            saved_items.append(item_dict)

            # Write to file after each item (for incremental saving)
            with open(self.save_file, "w", encoding="utf-8") as f:
                json.dump(saved_items, f, ensure_ascii=False, indent=2)

        # Return MediaItem objects with identifiers
        result_items = []
        for i, item_data in enumerate(saved_items):
            result_items.append(MediaItem(**item_data))

        return result_items
