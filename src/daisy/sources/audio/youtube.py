"""
YouTube audio source collection.
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


class YouTubeAudioSource(AudioSource):
    def __init__(
        self,
        save_file: str,
        language: str,
        overwrite: bool = False,
        **filters: Any,
    ):
        """
        Initialize the YouTube web source and filters.
        Filters:
        - "views": tuple[int, int] with min and max views (None for no filter)
        - "duration": tuple[int, int] with min and max duration (None for no filter)
        - "date": tuple[datetime, datetime] with min and max date (None for no filter) - this is both included in the query and in post-query filtering
        - "duration": tuple[int, int] with min and max duration in seconds (None for no filter) - this is only included in post-query filtering
        - "include_shorts": bool with True to include shorts in the results (this is a query modifier)
        - "section_length": int with the length of the section in the middle of the video to download in seconds (None for full video)
        """
        super().__init__(save_file, language, overwrite, **filters)
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
        # options.add_argument("--headless")
        self.driver = webdriver.Firefox(options=options)

    @exponential_backoff_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _navigate_with_retry(self, url: str) -> None:
        """Navigate to URL with exponential backoff retry"""
        self.driver.get(url)

    def parse_views(self, views: str) -> int:
        try:
            multiplier = 1
            views = views.replace(" views", "").replace(" view", "").strip()
            if views.endswith("K"):
                multiplier = 10**3
            elif views.endswith("M"):
                multiplier = 10**6
            elif views.endswith("B"):
                multiplier = 10**9
            float_views = float(
                views.replace("K", "").replace("M", "").replace("B", "")
            )
            return int(float_views * multiplier)
        except ValueError:
            return -1

    def parse_duration(self, duration: str) -> int:
        hours = 0
        minutes = 0
        seconds = 0
        if duration == "SHORTS":
            return -1
        split_duration = duration.split(":")
        try:
            if len(split_duration) == 3:
                hours = int(split_duration[0])
                minutes = int(split_duration[1])
                seconds = int(split_duration[2])
            elif len(split_duration) == 2:
                minutes = int(split_duration[0])
                seconds = int(split_duration[1])
            elif len(split_duration) == 1:
                seconds = int(split_duration[0])
        except ValueError:
            print(f"Error parsing duration: {duration}")
            return -1
        return hours * 3600 + minutes * 60 + seconds

    def parse_date(self, date: str) -> int:
        date = date.replace("Streamed", "").strip()
        return dateparser.parse(date)

    def search(self, media: MediaItem) -> list[AudioItem]:
        query = f"{media.name}"
        # remove bracketed content in the query, e.g. SomeContent (Name of the Content in English)
        query = re.sub(r"\(.*?\)", "", query)
        query = query.strip()
        if hasattr(self.filters, "date"):
            if self.filters.date[0] is not None:
                after_date = self.filters.date[0].strftime("%Y-%m-%d")
                query += f" after:{after_date}"
            if self.filters.date[1] is not None:
                before_date = self.filters.date[1].strftime("%Y-%m-%d")
                query += f" before:{before_date}"
        query = urllib.parse.quote(query)
        self._navigate_with_retry(
            f"https://www.youtube.com/results?search_query={query}"
        )
        # consent_button_xpath = "//button[contains(@class,'yt-spec-button-shape-next yt-spec-button-shape-next--filled')]"
        # WebDriverWait(self.driver, 20).until(
        #     EC.element_to_be_clickable((By.XPATH, consent_button_xpath))
        # ).click()
        try:
            WebDriverWait(self.driver, 4).until(
                EC.visibility_of_any_elements_located(
                    (
                        By.XPATH,
                        "//div[@id='contents']/ytd-video-renderer[@class='style-scope ytd-item-section-renderer']",
                    )
                )
            )
            time.sleep(0.3)
        except TimeoutException:
            print("Timeout waiting for video elements")
            return []
        data = self.driver.find_elements(
            By.XPATH,
            "//div[@id='contents']/ytd-video-renderer[@class='style-scope ytd-item-section-renderer']",
        )
        new_data = []
        for item in data:
            title = item.find_element(By.XPATH, ".//a[@id='video-title']").text
            try:
                views = item.find_elements(
                    By.XPATH,
                    ".//span[@class='inline-metadata-item style-scope ytd-video-meta-block']",
                )[0].text
                date = item.find_elements(
                    By.XPATH,
                    ".//span[@class='inline-metadata-item style-scope ytd-video-meta-block']",
                )[1].text
                duration = item.find_element(
                    By.XPATH,
                    ".//badge-shape",
                ).get_attribute("innerText")
            except (IndexError, NoSuchElementException):
                print(f"No views,date or duration element found, skipping... {title}")
                continue
            url = (
                item.find_element(By.XPATH, ".//a[@id='video-title']")
                .get_attribute("href")
                .split("&")[0]
            )
            channel_name = item.find_element(
                By.XPATH,
                ".//yt-formatted-string[@class='style-scope ytd-channel-name']",
            ).text
            new_data.append(
                {
                    "title": title,
                    "views": views,
                    "date": date,
                    "duration": duration,
                    "url": url,
                    "channel_name": channel_name,
                    "date_searched": datetime.now(),
                }
            )
        parsed_data = []
        for item in new_data:
            item["parsed_views"] = self.parse_views(item["views"])
            item["parsed_duration"] = self.parse_duration(item["duration"])
            item["parsed_date"] = self.parse_date(item["date"])
            if "shorts" not in item["url"]:
                item["url_id"] = re.search(r"v=([^&]+)", item["url"]).group(1)
            else:
                item["url_id"] = re.search(r"shorts/([^&]+)", item["url"]).group(1)
            item["identifier"] = f"{media.identifier}-{item['url_id']}"
            item["media_item_id"] = media.identifier
            parsed_data.append(item)
        # convert to AudioItem
        parsed_data = [AudioItem(**item) for item in parsed_data]
        for item in parsed_data:
            item.media_item_id = media.identifier
        return parsed_data

    def filter(self, items: list[AudioItem]) -> list[AudioItem]:
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
                date[0] is None or item.parsed_date > date[0],
                date[1] is None or item.parsed_date < date[1],
                include_shorts or item.parsed_duration != -1,
            ]
            if all(included):
                filtered_data.append(item)
        return filtered_data

    def close(self) -> None:
        if hasattr(self, "driver"):
            self.driver.quit()
