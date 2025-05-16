import os
from pathlib import Path
import json
from time import sleep
import argparse

import requests
from geopy import geocoders
from dotenv import load_dotenv

gn = geocoders.GeoNames(username=os.getenv("GEONAMES_USERNAME"))

# dotenv


load_dotenv()

YOUTUBE_API_KEYS = os.getenv("YOUTUBE_API_KEYS").split(",")

location_latlon = {
    "San Francisco, CA": (37.774929, -122.419416),
    "Madrid, ES": (40.416775, -3.70379),
    "Rome, IT": (41.903323, 12.45338),
    "Tokyo, JP": (35.681236, 139.767125),
    "Warsaw, PL": (52.229676, 21.012229),
    "Sao Paulo, BR": (-23.55052, -46.633309),
    "Istanbul, TR": (41.008238, 28.978359),
    "Shanghai, CN": (31.230416, 121.473701),
    "Paris, FR": (48.856613, 2.352222),
    "Berlin, DE": (52.520008, 13.404954),
    "Seoul, KR": (37.566535, 126.978371),
    "Riyadh, SA": (24.68773, 46.716539),
    "Moscow, RU": (55.755825, 37.617298),
    "Amsterdam, NL": (52.372776, 4.892222),
    "Mumbai, IN": (19.075984, 72.877656),
}


def get_videos(search_term, location, region_code, language, api_key):
    if location not in location_latlon:
        location = gn.geocode(location)
        location_latlon[location] = (location[1][0], location[1][1])
    location = location_latlon[location]
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "maxResults": 50,
        "q": search_term,
        "key": api_key,
        "location": f"{location[0]},{location[1]}",
        "locationRadius": "200km",
        "type": "video",
        "regionCode": region_code,
        "relevanceLanguage": language,
        "publishedAfter": "2025-01-01T00:00:00Z",
        "videoDuration": "long",
        "order": "viewCount",
    }
    response = requests.get(url, params=params, timeout=10)
    return response.json()


language_dict = {
    "en": ("San Francisco, CA", "US", "en"),
    "es": ("Madrid, ES", "ES", "es"),
    "it": ("Rome, IT", "IT", "it"),
    "ja": ("Tokyo, JP", "JP", "ja"),
    "pl": ("Warsaw, PL", "PL", "pl"),
    "pt": ("Sao Paulo, BR", "BR", "pt"),
    "tr": ("Istanbul, TR", "TR", "tr"),
    "zh": ("Shanghai, CN", "CN", "zh"),
    "fr": ("Paris, FR", "FR", "fr"),
    "de": ("Berlin, DE", "DE", "de"),
    "ko": ("Seoul, KR", "KR", "ko"),
    "ar": ("Riyadh, SA", "SA", "ar"),
    "ru": ("Moscow, RU", "RU", "ru"),
    "nl": ("Amsterdam, NL", "NL", "nl"),
    "hi": ("Mumbai, IN", "IN", "hi"),
}

search_terms = {
    "en": [
        "talk show",
        "interview",
        "debate",
        "sports commentary",
        "news",
        "politics",
        "economy",
        "technology",
        "science",
        "podcast",
    ],
    "es": [
        "programa de entrevistas",
        "entrevista",
        "debate",
        "comentario deportivo",
        "noticias",
        "política",
        "economía",
        "tecnología",
        "ciencia",
        "podcast",
    ],
    "it": [
        "talk show",
        "intervista",
        "dibattito",
        "commento sportivo",
        "notizie",
        "politica",
        "economia",
        "tecnologia",
        "scienza",
        "podcast",
    ],
    "ja": [
        "トークショー",
        "インタビュー",
        "討論",
        "スポーツ解説",
        "ニュース",
        "政治",
        "経済",
        "テクノロジー",
        "科学",
        "ポッドキャスト",
    ],
    "pl": [
        "talk-show",
        "wywiad",
        "debata",
        "komentarz sportowy",
        "wiadomości",
        "polityka",
        "gospodarka",
        "technologia",
        "nauka",
        "podcast",
    ],
    "pt": [
        "talk show",
        "entrevista",
        "debate",
        "comentário esportivo",
        "notícias",
        "política",
        "economia",
        "tecnologia",
        "ciência",
        "podcast",
    ],
    "tr": [
        "talk show",
        "röportaj",
        "tartışma",
        "spor yorumu",
        "haberler",
        "politika",
        "ekonomi",
        "teknoloji",
        "bilim",
        "podcast",
    ],
    "zh": [
        "脱口秀",
        "采访",
        "辩论",
        "体育评论",
        "新闻",
        "政治",
        "经济",
        "科技",
        "科学",
        "播客",
    ],
    "fr": [
        "talk-show",
        "entretien",
        "débat",
        "commentaire sportif",
        "actualités",
        "politique",
        "économie",
        "technologie",
        "science",
        "podcast",
    ],
    "de": [
        "Talkshow",
        "Interview",
        "Debatte",
        "Sportkommentar",
        "Nachrichten",
        "Politik",
        "Wirtschaft",
        "Technologie",
        "Wissenschaft",
        "Podcast",
    ],
    "ko": [
        "토크쇼",
        "인터뷰",
        "토론",
        "스포츠 해설",
        "뉴스",
        "정치",
        "경제",
        "기술",
        "과학",
        "팟캐스트",
    ],
    "ar": [
        "برنامج حواري",
        "مقابلة",
        "مناظرة",
        "تعليق رياضي",
        "أخبار",
        "سياسة",
        "اقتصاد",
        "تكنولوجيا",
        "علوم",
        "بودكاست",
    ],
    "ru": [
        "ток-шоу",
        "интервью",
        "дебаты",
        "спортивный комментарий",
        "новости",
        "политика",
        "экономика",
        "технологии",
        "наука",
        "подкаст",
    ],
    "nl": [
        "talkshow",
        "interview",
        "debat",
        "sportcommentaar",
        "nieuws",
        "politiek",
        "economie",
        "technologie",
        "wetenschap",
        "podcast",
    ],
    "hi": [
        "टॉक शो",
        "साक्षात्कार",
        "बहस",
        "खेल टिप्पणी",
        "समाचार",
        "राजनीति",
        "अर्थव्यवस्था",
        "प्रौद्योगिकी",
        "विज्ञान",
        "पॉडकास्ट",
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get YouTube videos by language and search term"
    )
    parser.add_argument(
        "--language_code",
        type=str,
        help="Specific language code to process (e.g. 'en', 'es')",
    )
    args = parser.parse_args()

    current_key_index = 0
    stop = False

    # If language_code is provided, only process that language
    languages_to_process = (
        [args.language_code] if args.language_code else language_dict.keys()
    )

    for search_term_idx in range(len(search_terms["en"])):
        for language in languages_to_process:
            if language not in language_dict:
                print(f"Invalid language code: {language}")
                continue

            lang_tuple = language_dict[language]
            if stop:
                break
            if Path(
                f"videos/{language}/{search_terms[language][search_term_idx].replace(' ', '_')}.json"
            ).exists():
                print(
                    f"Skipping {search_terms[language][search_term_idx]} in {language} because it already exists"
                )
                continue
            Path(f"videos/{language}").mkdir(parents=True, exist_ok=True)
            print(
                f"Getting videos for {search_terms[language][search_term_idx]} in {language}"
            )
            videos = get_videos(
                search_terms[language][search_term_idx],
                lang_tuple[0],
                lang_tuple[1],
                lang_tuple[2],
                YOUTUBE_API_KEYS[current_key_index],
            )
            if "error" in videos and videos["error"]["code"] == 403:
                current_key_index += 1
                if current_key_index >= len(YOUTUBE_API_KEYS):
                    print(f"No more API keys available, stopping")
                    stop = True
                    break
                videos = get_videos(
                    search_terms[language][search_term_idx],
                    lang_tuple[0],
                    lang_tuple[1],
                    lang_tuple[2],
                    YOUTUBE_API_KEYS[current_key_index],
                )
                if "error" in videos and videos["error"]["code"] == 403:
                    raise Exception(
                        f"API key {YOUTUBE_API_KEYS[current_key_index]} is blocked"
                    )
            with open(
                f"videos/{language}/{search_terms[language][search_term_idx].replace(' ', '_')}.json",
                "w",
                encoding="utf-8",
            ) as f:
                # dump formatted
                json.dump(videos, f, indent=4)
            sleep(1)
