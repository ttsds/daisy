import os
from json import JSONDecodeError

import openai

from daisy.abstract import (
    FilterResult,
    FilterResultList,
    MediaItem,
    AudioItem,
    LANGUAGES,
)


class ResultsFilter:

    def __init__(
        self,
        llm_id: str = "x-ai/grok-4-fast",
    ):
        self.llm_id = llm_id
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_KEY"),
        )

    def filter(self, source: MediaItem, items: list[AudioItem]) -> list[FilterResult]:
        shortened_items = []
        for item in items:
            shortened_items.append(
                {
                    "identifier": item.identifier,
                    "title": item.title,
                    "channel_name": item.channel_name,
                }
            )
        prompt = f"""
        Filter the following audio items by returning a list of FilterResult objects, which have the following fields:
        - identifier: The identifier of the audio item.
        - is_from_creator: Whether the audio item is from the creator of the source.
        - is_right_language: Whether the audio item is in the right language.
        - mostly_spoken_content: Whether the audio of the item is mostly spoken content. For example, music videos are not spoken content.
        - category: The category of the audio item.
        The source is {source.name} and has the following description: {source.description}
        The language is {LANGUAGES[source.language].english_name} ({LANGUAGES[source.language].native_name})
        The proposed categories are {",".join(source.categories)} but you can change them if you think they are not correct.
        
        Here are the audio items:
        {shortened_items}
        """
        max_tries = 3
        for i in range(max_tries):
            try:
                print(prompt)
                response = self.client.chat.completions.parse(
                    model=self.llm_id,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=FilterResultList,
                )
                results = response.choices[0].message.parsed
                # sort results by how many flags are true
                results.items.sort(
                    key=lambda x: sum(
                        [
                            x.is_from_creator,
                            x.is_right_language,
                            x.mostly_spoken_content,
                        ]
                    ),
                    reverse=True,
                )
                return results
            except JSONDecodeError as e:
                print(f"Error filtering results: {e}")
                continue
            except AttributeError as e:
                print(f"Error filtering results: {e}")
                continue
        raise Exception("Failed to filter results")
