"""
Bilibili audio source collection.
"""

import re
import urllib.parse
from types import SimpleNamespace
from typing import Any
import dateparser
import time
import random
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)

from daisy.core import AudioSource, MediaItem, AudioItem
from daisy.utils.retry import exponential_backoff_retry


class BilibiliAudioSource(AudioSource):
    def __init__(
        self,
        save_file: str,
        language: str,
        overwrite: bool = False,
        **filters: Any,
    ):
        """
        Initialize the Bilibili web source and filters.
        Filters:
        - "views": tuple[int, int] with min and max views (None for no filter)
        - "duration": tuple[int, int] with min and max duration (None for no filter)
        - "date": tuple[datetime, datetime] with min and max date (None for no filter) - applied in post-query filtering
        - "include_shorts": bool with True to include shorts in the results
        - "section_length": int with the length of the section in the middle of the video to download in seconds (None for full video)
        """
        super().__init__(save_file, language, overwrite, **filters)

        self.base_url = "https://api.bilibili.com"

        default_filters = {
            "views": (None, None),
            "duration": (None, None),
            "date": (None, None),
            "include_shorts": False,
            "section_length": None,
        }
        self.filters = default_filters.copy()
        self.filters.update(filters)
        self.filters = SimpleNamespace(**self.filters)

    def init_driver(self) -> None:
        options = webdriver.FirefoxOptions()
        # options.add_experimental_option(
        #     "prefs", {"intl.accept_languages": self.language.iso2}
        # )
        self.driver = webdriver.Firefox(options=options)

    @exponential_backoff_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _navigate_with_retry(self, url: str) -> None:
        """Navigate to URL with exponential backoff retry"""
        self.driver.get(url)

    def parse_views(self, views: str) -> int:
        """Parse Bilibili view count format (e.g., '1.2万', '3.4万', '1.2亿')"""
        if not views or views == "0":
            return 0

        views = views.replace("播放", "").replace("次", "").strip()

        multiplier = 1
        if "万" in views:
            multiplier = 10**4
        elif "亿" in views:
            multiplier = 10**8
        elif "千" in views:
            multiplier = 10**3

        # Remove Chinese characters and convert to float
        numeric_part = re.sub(r"[^\d.]", "", views)
        if not numeric_part:
            return 0

        try:
            float_views = float(numeric_part)
            return int(float_views * multiplier)
        except ValueError:
            return 0

    def parse_duration(self, duration: str) -> int:
        """Parse Bilibili duration format (e.g., '3:45', '1:23:45')"""
        if not duration:
            return 0  # Empty string
        if duration == "直播":
            return -1  # Live stream

        try:
            parts = duration.split(":")
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            else:
                return int(parts[0])  # Just seconds
        except (ValueError, IndexError):
            return 0

    def parse_date(self, date_str: str) -> int:
        """Parse Bilibili date format"""
        date_str = date_str.replace("·", "").strip()
        try:
            return dateparser.parse(date_str)
        except Exception:
            print(f"Error parsing date: {date_str}")
            return None

    def search(self, media: MediaItem) -> list[AudioItem]:
        """Search for videos on Bilibili"""
        query = media.name
        query = re.sub(r"\(.*?\)", "", query)
        query = query.strip()

        # Encode query for URL
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://search.bilibili.com/video?keyword={encoded_query}"

        # example pubtime_begin_s=1759096800&pubtime_end_s=1760133599
        if self.filters.date[0] is not None:
            search_url += f"&pubtime_begin_s={int(self.filters.date[0].timestamp())}"
        if self.filters.date[1] is not None:
            search_url += f"&pubtime_end_s={int(self.filters.date[1].timestamp())}"

        # Navigate to search page
        self._navigate_with_retry(search_url)

        # Wait for page to load
        try:
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "bili-video-card__wrap"))
            )
            time.sleep(0.3)
        except TimeoutException:
            print("Timeout waiting for video elements")
            return []

        # Find video items
        video_items = self.driver.find_elements(
            By.XPATH, "//div[@class='bili-video-card__wrap']"
        )

        data = []
        for item in video_items:
            # try:
            # Extract video information
            title_element = item.find_element(
                By.XPATH, ".//h3[@class='bili-video-card__info--tit']"
            )
            title = title_element.get_attribute("title")
            # find parent of title_element and get href
            url = title_element.find_element(By.XPATH, "./..").get_attribute("href")
            # Extract views
            views_element = item.find_elements(
                By.XPATH, ".//div[@class='bili-video-card__stats--left']/span"
            )[0]
            views = views_element.get_attribute("innerText")

            # Extract duration
            duration_element = item.find_element(
                By.XPATH, ".//span[@class='bili-video-card__stats__duration']"
            )
            duration = duration_element.text.strip()

            # Extract upload date
            try:
                date_element = item.find_element(
                    By.XPATH, ".//span[@class='bili-video-card__info--date']"
                )
                date = date_element.text.strip()
            except NoSuchElementException:
                print(f"No date element found, skipping... {title}")
                continue

            # Extract channel name
            channel_element = item.find_element(
                By.XPATH, ".//span[@class='bili-video-card__info--author']"
            )
            channel_name = channel_element.text.strip()

            # Extract video ID from URL
            url_id = re.search(r"/video/([^/?]+)", url)
            if url_id:
                url_id = url_id.group(1)
            else:
                url_id = url.split("/")[-1]

            data.append(
                {
                    "title": title,
                    "views": views,
                    "date": date,
                    "duration": duration,
                    "url": url,
                    "channel_name": channel_name,
                    "url_id": url_id,
                    "date_searched": datetime.now(),
                }
            )

        # except Exception as e:
        #     raise e

        # Parse the data
        parsed_data = []
        for item in data:
            item["parsed_views"] = self.parse_views(item["views"])
            item["parsed_duration"] = self.parse_duration(item["duration"])
            item["parsed_date"] = self.parse_date(item["date"])
            parsed_data.append(item)

        # Convert to AudioItem objects
        audio_items = []
        for item in parsed_data:
            audio_item = AudioItem(
                identifier=item["url_id"],
                title=item["title"],
                views=item["views"],
                date=item["date"],
                duration=item["duration"],
                url=item["url"],
                channel_name=item["channel_name"],
                url_id=item["url_id"],
                parsed_views=item["parsed_views"],
                parsed_duration=item["parsed_duration"],
                parsed_date=item["parsed_date"],
                media_item_id=media.identifier,
            )
            audio_items.append(audio_item)

        return audio_items

    def filter(self, items: list[AudioItem]) -> list[AudioItem]:
        """Filter audio items based on specified criteria"""
        filtered_data = []
        for item in items:
            views = getattr(self.filters, "views", (None, None))
            duration = getattr(self.filters, "duration", (None, None))
            date = getattr(self.filters, "date", (None, None))
            include_shorts = getattr(self.filters, "include_shorts", False)

            included = [
                views[0] is None or item.parsed_views > views[0],
                views[1] is None or item.parsed_views < views[1],
                duration[0] is None or item.parsed_duration > duration[0],
                duration[1] is None or item.parsed_duration < duration[1],
                date[0] is None
                or (item.parsed_date is not None and item.parsed_date > date[0]),
                date[1] is None
                or (item.parsed_date is not None and item.parsed_date < date[1]),
                include_shorts or item.parsed_duration != -1,  # -1 for live streams
            ]

            if all(included):
                filtered_data.append(item)

        return filtered_data

    def close(self) -> None:
        if hasattr(self, "driver"):
            self.driver.quit()
