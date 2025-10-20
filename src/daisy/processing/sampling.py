"""
Sampling and language detection for audio items.
"""

from typing import List, Optional

import pandas as pd
import numpy as np
import fasttext
from huggingface_hub import hf_hub_download
import random

from daisy.core import FilterResult, MediaItem, AudioItem, FullItem, LANGUAGES

# map for non-standard fasttext language codes
FASTTEXT_MAP = {
    "als": "sq",
    "zh": "cmn",
    "arb": "ara",
    "azj": "aze",
    "zho": "cmn",
    "tgl": "fil",
    "lvs": "lv",
    "zsm": "ms",
    "npi": "ne",
    "nob": "no",
    "nno": "nn",
    "ory": "or",
    "pes": "fa",
    "swh": "sw",
}


class ResultSampler:
    def __init__(
        self,
        media: List[MediaItem],
        audio: List[AudioItem],
        filter_results: List[FilterResult],
    ):
        self.media = media
        self.audio = audio
        self.filter_results = filter_results
        # create dataframe with media item, audio item, and filter result joined by identifier
        media_names = [media.name for media in media]
        media_ids = [media.identifier for media in media]
        media_descriptions = [media.description for media in media]
        media_categories = [media.categories for media in media]
        media_languages = [media.language for media in media]
        media_countries = [media.country for media in media]
        media_types = [media.identifier.split("-")[1] for media in media]
        audio_titles = [audio.title for audio in audio]
        audio_identifiers = [audio.identifier for audio in audio]
        audio_channels = [audio.channel_name for audio in audio]
        audio_urls = [audio.url for audio in audio]
        audio_views = [audio.parsed_views for audio in audio]
        audio_dates = [audio.parsed_date for audio in audio]
        audio_durations = [audio.parsed_duration for audio in audio]
        audio_media_ids = [audio.media_item_id for audio in audio]
        filter_results_identifiers = [
            filter_result.identifier for filter_result in filter_results
        ]
        filter_results_is_from_creators = [
            filter_result.is_from_creator for filter_result in filter_results
        ]
        filter_results_is_right_languages = [
            filter_result.is_right_language for filter_result in filter_results
        ]
        filter_results_mostly_spoken_contents = [
            filter_result.mostly_spoken_content for filter_result in filter_results
        ]
        filter_results_categories = [
            filter_result.category for filter_result in filter_results
        ]
        media_df = pd.DataFrame(
            {
                "name": media_names,
                "media_item_id": media_ids,
                "media_type": media_types,
                "description": media_descriptions,
                "categories": media_categories,
                "language": media_languages,
                "country": media_countries,
            }
        )
        audio_df = pd.DataFrame(
            {
                "title": audio_titles,
                "audio_item_id": audio_identifiers,
                "channel_name": audio_channels,
                "url": audio_urls,
                "views": audio_views,
                "date": audio_dates,
                "duration": audio_durations,
                "media_item_id": audio_media_ids,
            }
        )
        filter_results_df = pd.DataFrame(
            {
                "audio_item_id": filter_results_identifiers,
                "is_from_creator": filter_results_is_from_creators,
                "is_right_language": filter_results_is_right_languages,
                "mostly_spoken_content": filter_results_mostly_spoken_contents,
                "category": filter_results_categories,
            }
        )
        self.df = media_df.merge(audio_df, on="media_item_id").merge(
            filter_results_df, on="audio_item_id"
        )
        self.df_filtered = self.df[
            self.df["is_right_language"]
            & self.df["mostly_spoken_content"]
            & self.df["is_from_creator"]
        ]
        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification",
            filename="model.bin",
            local_dir="models",
        )
        self.fasttext_model = fasttext.load_model(model_path)
        # filter out rows where the language does not match the media language
        language_detected = []
        detected_languages = []
        for _, row in self.df_filtered.iterrows():
            # print(row["title"])
            prediction = self.fasttext_model.predict(row["title"])[0][0]
            prediction = prediction.split("__")[-1].split("_")[0]
            if prediction in FASTTEXT_MAP:
                prediction = FASTTEXT_MAP[prediction]
            detected_languages.append(prediction)
            # fasttext does not seem to work for serbian, so for this we allow bos, hrv, slv and pol
            if row["language"] == "sr":
                language_detected.append(prediction in ["bos", "hrv", "slv", "pol"])
            else:
                language_detected.append(
                    (
                        prediction in LANGUAGES[row["language"]].iso3
                        or (
                            len(prediction) == 2
                            and prediction in LANGUAGES[row["language"]].iso2
                        )
                    )
                )
        self.df_filtered = self.df_filtered[language_detected]

    def sample(
        self, n: int, return_df: bool = False
    ) -> tuple[List[FullItem], Optional[pd.DataFrame]]:
        """
        sample without replacement, trying to balance media types and biasing towards higher durations, view counts and is_from_creator
        """
        if n <= 0:
            if return_df:
                return [], pd.DataFrame()
            return []

        if len(self.df_filtered) == 0:
            if return_df:
                return [], pd.DataFrame()
            return []

        # Calculate sampling weights based on the criteria
        weights = []
        for _, row in self.df_filtered.iterrows():
            weight = 1.0

            # Bias towards higher durations (normalize to 0-1 range)
            if pd.notna(row["duration"]) and row["duration"] > 0:
                duration_weight = min(row["duration"] / 3600, 1.0)  # Cap at 1 hour
                weight += duration_weight * 2  # 2x weight for duration

            # Bias towards higher view counts (log scale to avoid extreme values)
            if pd.notna(row["views"]) and row["views"] > 0:
                view_weight = min(
                    np.log10(row["views"] + 1) / 6, 1.0
                )  # Cap at 1M views
                weight += view_weight * 1.5  # 1.5x weight for views

            # Bias towards creator content
            if row["is_from_creator"]:
                weight += 1.0  # Additional weight for creator content

            weights.append(weight)

        # Convert to numpy array for easier handling
        weights = np.array(weights)

        # Get unique media names for balancing
        media_names = self.df_filtered["name"].unique()

        # Calculate how many samples per media name (roughly balanced)
        samples_per_name = max(1, n // len(media_names))
        remaining_samples = n % len(media_names)

        sampled_indices = []
        available_indices = set(range(len(self.df_filtered)))

        # Sample from each media name
        for i, media_name in enumerate(media_names):
            name_mask = self.df_filtered["name"] == media_name
            name_indices = np.where(name_mask)[0]

            # Filter to only available indices
            available_name_indices = [
                idx for idx in name_indices if idx in available_indices
            ]

            if not available_name_indices:
                continue

            # Calculate how many samples for this name
            num_samples = samples_per_name
            if i < remaining_samples:
                num_samples += 1

            # Don't sample more than available
            num_samples = min(num_samples, len(available_name_indices))

            if num_samples > 0:
                # Get weights for this media name
                name_weights = weights[available_name_indices]

                # Normalize weights to probabilities
                name_probs = name_weights / name_weights.sum()

                # Sample without replacement
                sampled_name_indices = np.random.choice(
                    available_name_indices,
                    size=num_samples,
                    replace=False,
                    p=name_probs,
                )

                sampled_indices.extend(sampled_name_indices)
                available_indices -= set(sampled_name_indices)

        # If we still need more samples, fill from remaining available items
        if len(sampled_indices) < n and available_indices:
            remaining_needed = n - len(sampled_indices)
            remaining_indices = list(available_indices)

            if remaining_indices:
                remaining_weights = weights[remaining_indices]
                remaining_probs = remaining_weights / remaining_weights.sum()

                additional_samples = min(remaining_needed, len(remaining_indices))
                additional_indices = np.random.choice(
                    remaining_indices,
                    size=additional_samples,
                    replace=False,
                    p=remaining_probs,
                )

                sampled_indices.extend(additional_indices)

        # Convert sampled indices to FullItem objects
        sampled_items = []
        for idx in sampled_indices:
            row = self.df_filtered.iloc[idx]

            # Find the corresponding objects
            media_item = next(
                m for m in self.media if m.identifier == row["media_item_id"]
            )
            audio_item = next(
                a for a in self.audio if a.identifier == row["audio_item_id"]
            )
            filter_result = next(
                f for f in self.filter_results if f.identifier == row["audio_item_id"]
            )

            full_item = FullItem(
                media_item=media_item,
                audio_item=audio_item,
                filter_result=filter_result,
            )
            sampled_items.append(full_item)

        if len(sampled_items) > n:
            # seed
            random.seed(42)
            sampled_items = random.sample(sampled_items, n)

        if not return_df:
            return sampled_items
        else:
            media_names = []
            media_ids = []
            media_types = []
            media_descriptions = []
            media_categories = []
            media_languages = []
            media_countries = []
            audio_titles = []
            audio_identifiers = []
            audio_channels = []
            audio_urls = []
            audio_views = []
            audio_dates = []
            audio_durations = []
            audio_media_ids = []
            filter_results_identifiers = []
            filter_results_is_from_creators = []
            filter_results_is_right_languages = []
            filter_results_mostly_spoken_contents = []
            filter_results_categories = []
            for item in sampled_items:
                media_names.append(item.media_item.name)
                media_ids.append(item.media_item.identifier)
                media_types.append(item.media_item.identifier.split("-")[1])
                media_descriptions.append(item.media_item.description)
                media_categories.append(item.media_item.categories)
                media_languages.append(item.media_item.language)
                media_countries.append(item.media_item.country)
                audio_titles.append(item.audio_item.title)
                audio_identifiers.append(item.audio_item.identifier)
                audio_channels.append(item.audio_item.channel_name)
                audio_urls.append(item.audio_item.url)
                audio_views.append(item.audio_item.parsed_views)
                audio_dates.append(item.audio_item.parsed_date)
                audio_durations.append(item.audio_item.parsed_duration)
                audio_media_ids.append(item.audio_item.media_item_id)
                filter_results_identifiers.append(item.filter_result.identifier)
                filter_results_is_from_creators.append(
                    item.filter_result.is_from_creator
                )
                filter_results_is_right_languages.append(
                    item.filter_result.is_right_language
                )
                filter_results_mostly_spoken_contents.append(
                    item.filter_result.mostly_spoken_content
                )
                filter_results_categories.append(item.filter_result.category)
            df = pd.DataFrame(
                {
                    "media_item_id": media_ids,
                    "audio_item_id": audio_identifiers,
                    "name": media_names,
                    "categories": media_categories,
                    "language": media_languages,
                    "country": media_countries,
                    "title": audio_titles,
                    "channel_name": audio_channels,
                    "url": audio_urls,
                    "views": audio_views,
                    "date": audio_dates,
                    "duration": audio_durations,
                    "category": media_types,
                }
            )
            # sort by audio_item_id
            df = df.sort_values(by="audio_item_id")
            return sampled_items, df
