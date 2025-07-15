import os
import subprocess
import random
import time
from pathlib import Path
import json
from time import sleep
import argparse
import re
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from geopy import geocoders
from dotenv import load_dotenv

gn = geocoders.GeoNames(username=os.getenv("GEONAMES_USERNAME"))

# dotenv
load_dotenv()

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


# Function to connect Windscribe VPN
def windscribe_connect(location):
    """
    Connect to Windscribe VPN at specified location.

    Args:
        location (str): The VPN location to connect to
    """
    print(f"\n🌐 Connecting Windscribe VPN to: {location}")
    subprocess.run(["windscribe-cli", "disconnect"], stdout=subprocess.DEVNULL)
    time.sleep(3)
    subprocess.run(["windscribe-cli", "connect", location], check=True)
    print("✅ VPN Connected.")
    time.sleep(5)  # Wait to ensure stable connection


def setup_driver():
    """Setup Chrome driver with appropriate options for YouTube scraping"""
    chrome_options = Options()
    # Removed headless mode to show browser window
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    driver = webdriver.Chrome(options=chrome_options)
    return driver


def extract_video_info(video_element):
    """Extract video information from a YouTube video element"""
    try:
        # Extract video title
        title_element = video_element.find_element(By.CSS_SELECTOR, "a#video-title")
        title = title_element.get_attribute("title") or title_element.text

        # Extract video URL
        video_url = title_element.get_attribute("href")
        if video_url and not video_url.startswith("http"):
            video_url = "https://www.youtube.com" + video_url

        # Extract video ID from URL
        video_id = None
        if video_url:
            video_id_match = re.search(r"watch\?v=([a-zA-Z0-9_-]+)", video_url)
            if video_id_match:
                video_id = video_id_match.group(1)

        # Extract channel name
        channel_element = video_element.find_element(By.CSS_SELECTOR, "a#channel-name")
        channel_name = channel_element.text.strip()

        # Extract view count and upload time
        metadata_element = video_element.find_element(
            By.CSS_SELECTOR, "div#metadata-line"
        )
        metadata_text = metadata_element.text

        # Extract view count
        view_count = None
        view_match = re.search(r"([\d,]+)\s*views?", metadata_text)
        if view_match:
            view_count = view_match.group(1).replace(",", "")

        # Extract upload time
        upload_time = None
        time_match = re.search(
            r"(\d+\s*(?:hour|day|week|month|year)s?\s*ago)", metadata_text
        )
        if time_match:
            upload_time = time_match.group(1)

        # Extract description
        description_element = video_element.find_element(
            By.CSS_SELECTOR, "div#description-text"
        )
        description = description_element.text.strip()

        return {
            "videoId": video_id,
            "title": title,
            "channelTitle": channel_name,
            "viewCount": view_count,
            "uploadTime": upload_time,
            "description": description,
            "url": video_url,
        }
    except Exception as e:
        print(f"Error extracting video info: {e}")
        return None


def get_videos_selenium(search_term, location, region_code, language, driver):
    """Get YouTube videos using Selenium instead of API"""
    try:
        # Construct YouTube search URL with filters
        search_url = f"https://www.youtube.com/results?search_query={search_term.replace(' ', '+')}&sp=CAI%253D"  # CAI%253D = sort by view count

        print(f"Searching for: {search_term}")
        driver.get(search_url)

        # Wait for videos to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-video-renderer"))
        )

        # Scroll to load more videos
        for _ in range(3):  # Scroll 3 times to load more content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(2)

        # Find all video elements
        video_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer")

        videos = []
        for video_element in video_elements[:50]:  # Limit to 50 videos
            video_info = extract_video_info(video_element)
            if video_info:
                videos.append(
                    {
                        "kind": "youtube#searchResult",
                        "etag": "dummy_etag",
                        "id": {
                            "kind": "youtube#video",
                            "videoId": video_info["videoId"],
                        },
                        "snippet": {
                            "publishedAt": "2025-01-01T00:00:00Z",  # We can't get exact publish date from search
                            "channelId": "dummy_channel_id",
                            "title": video_info["title"],
                            "description": video_info["description"],
                            "thumbnails": {
                                "default": {
                                    "url": f"https://img.youtube.com/vi/{video_info['videoId']}/default.jpg"
                                },
                                "medium": {
                                    "url": f"https://img.youtube.com/vi/{video_info['videoId']}/mqdefault.jpg"
                                },
                                "high": {
                                    "url": f"https://img.youtube.com/vi/{video_info['videoId']}/hqdefault.jpg"
                                },
                            },
                            "channelTitle": video_info["channelTitle"],
                            "liveBroadcastContent": "none",
                            "publishTime": "2025-01-01T00:00:00Z",
                        },
                        "statistics": {"viewCount": video_info["viewCount"]},
                    }
                )

        return {
            "kind": "youtube#searchListResponse",
            "etag": "dummy_etag",
            "nextPageToken": None,
            "prevPageToken": None,
            "pageInfo": {"totalResults": len(videos), "resultsPerPage": len(videos)},
            "items": videos,
        }

    except TimeoutException:
        print(f"Timeout waiting for videos to load for search term: {search_term}")
        return {"error": {"code": 408, "message": "Request timeout"}}
    except Exception as e:
        print(f"Error getting videos for {search_term}: {e}")
        return {"error": {"code": 500, "message": str(e)}}


# Define language-to-Windscribe locations
language_dict = {
    "en": ("San Francisco", "US", "en"),
    "es": ("Madrid", "ES", "es"),
    "it": ("Rome", "IT", "it"),
    "ja": ("Tokyo", "JP", "ja"),
    "pl": ("Warsaw", "PL", "pl"),
    "pt": ("Sao Paulo", "BR", "pt"),
    "tr": ("Istanbul", "TR", "tr"),
    "zh": ("Hong Kong", "HK", "zh"),
    "fr": ("Paris", "FR", "fr"),
    "de": ("Frankfurt", "DE", "de"),
    "ko": ("Seoul", "KR", "ko"),
    "ar": ("Dubai", "AE", "ar"),
    "ru": ("Moscow", "RU", "ru"),
    "nl": ("Amsterdam", "NL", "nl"),
    "hi": ("Mumbai", "IN", "hi"),
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
        "vlog",
        "tutorial",
        "product review",
        "unboxing",
        "gaming",
        "gameplay",
        "documentary",
        "comedy",
        "lecture",
        "storytime",
    ],
    "es": [
        "programa de entrevistas",
        "talk show",
        "entrevista",
        "debate",
        "comentario deportivo",
        "noticias",
        "política",
        "economía",
        "tecnología",
        "ciencia",
        "podcast",
        "vlog",
        "tutorial",
        "reseña de producto",
        "gameplay",
        "documental",
        "comedia",
        "conferencia",
        "storytime",
        "contando mi historia",
    ],
    "it": [
        "talk show",
        "intervista",
        "dibattito",
        "telecronaca",
        "notizie",
        "politica",
        "economia",
        "tecnologia",
        "scienza",
        "podcast",
        "vlog",
        "tutorial",
        "recensione",
        "unboxing",
        "gaming",
        "documentario",
        "commedia",
        "conferenza",
        "storytime",
        "la mia storia",
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
        "Vlog",
        "チュートリアル",
        "商品レビュー",
        "開封動画",
        "ゲーム実況",
        "ドキュメンタリー",
        "お笑い",
        "講演",
        "体験談",
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
        "vlog",
        "poradnik",
        "recenzja produktu",
        "unboxing",
        "zagrajmy w",
        "film dokumentalny",
        "komedia",
        "wykład",
        "historia z mojego życia",
    ],
    "pt": [
        "talk show",
        "entrevista",
        "debate",
        "comentário esportivo",
        "narração esportiva",
        "notícias",
        "política",
        "economia",
        "tecnologia",
        "ciência",
        "podcast",
        "vlog",
        "tutorial",
        "análise de produto",
        "review",
        "gameplay",
        "documentário",
        "comédia",
        "palestra",
        "storytime",
        "contando minha história",
    ],
    "tr": [
        "talk show",
        "röportaj",
        "tartışma",
        "münazara",
        "spor yorumu",
        "haberler",
        "politika",
        "ekonomi",
        "teknoloji",
        "bilim",
        "podcast",
        "vlog",
        "nasıl yapılır",
        "ürün incelemesi",
        "kutu açılımı",
        "oynanış",
        "belgesel",
        "komedi",
        "söyleşi",
        "yaşanmış hikayeler",
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
        "Vlog",
        "教程",
        "产品评测",
        "开箱",
        "游戏实况",
        "游戏解说",
        "纪录片",
        "喜剧",
        "相声",
        "讲座",
        "经历分享",
    ],
    "fr": [
        "talk-show",
        "interview",
        "débat",
        "commentaire sportif",
        "actualités",
        "politique",
        "économie",
        "technologie",
        "science",
        "podcast",
        "vlog",
        "tutoriel",
        "vulgarisation",
        "test de produit",
        "unboxing",
        "gameplay FR",
        "documentaire",
        "comédie",
        "conférence",
        "storytime",
        "mon histoire",
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
        "Vlog",
        "Anleitung",
        "Produkttest",
        "Unboxing",
        "Let's Play",
        "Doku",
        "Comedy",
        "Vortrag",
        "meine Geschichte",
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
        "브이로그",
        "튜토리얼",
        "제품 리뷰",
        "언박싱",
        "하울",
        "게임 방송",
        "다큐멘터리",
        "코미디",
        "강의",
        "썰",
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
        "فلوج",
        "شرح",
        "مراجعة منتج",
        "فتح صندوق",
        "جيم بلاي",
        "تختيم",
        "وثائقي",
        "كوميديا",
        "محاضرة",
        "سوالف",
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
        "влог",
        "туториал",
        "гайд",
        "обзор",
        "распаковка",
        "летсплей",
        "документалка",
        "стендап",
        "лекция",
        "история из жизни",
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
        "vlog",
        "uitleg",
        "productrecensie",
        "review",
        "unboxing",
        "Let's Play",
        "documentaire",
        "comedy",
        "lezing",
        "waargebeurd verhaal",
    ],
    "hi": [
        "टॉक शो",
        "इंटरव्यू",
        "बहस",
        "कमेंट्री",
        "समाचार",
        "राजनीति",
        "अर्थव्यवस्था",
        "प्रौद्योगिकी",
        "विज्ञान",
        "पॉडकास्ट",
        "व्लॉग",
        "ट्यूटोरियल",
        "प्रोडक्ट रिव्यू",
        "अनबॉक्सिंग",
        "गेमप्ले",
        "डॉक्यूमेंट्री",
        "कॉमेडी",
        "भाषण",
        "मेरी कहानी",
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get YouTube videos by language and search term using Selenium"
    )
    parser.add_argument(
        "--language_code",
        type=str,
        help="Specific language code to process (e.g. 'en', 'es')",
    )
    parser.add_argument(
        "--use_vpn",
        action="store_true",
        help="Use Windscribe VPN for location-based scraping",
    )
    args = parser.parse_args()

    # Setup Selenium driver
    driver = setup_driver()

    try:
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

                # Connect VPN aligned with language location
                if args.use_vpn:
                    windscribe_location = f"{lang_tuple[0]}"
                    windscribe_connect(windscribe_location)

                output_file = f"videos/{language}/{search_terms[language][search_term_idx].replace(' ', '_')}.json"

                if Path(output_file).exists():
                    print(
                        f"Skipping {search_terms[language][search_term_idx]} in {language} because it already exists"
                    )
                    continue

                Path(f"videos/{language}").mkdir(parents=True, exist_ok=True)
                print(
                    f"Getting videos for {search_terms[language][search_term_idx]} in {language}"
                )

                videos = get_videos_selenium(
                    search_terms[language][search_term_idx],
                    lang_tuple[0],
                    lang_tuple[1],
                    lang_tuple[2],
                    driver,
                )

                if "error" in videos:
                    print(f"Error getting videos: {videos['error']}")
                    continue

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(videos, f, indent=4)

                sleep(2)  # Be respectful to YouTube's servers

    finally:
        # Disconnect VPN after completion
        if args.use_vpn:
            subprocess.run(["windscribe-cli", "disconnect"], check=True)
            print("\n🚩 All scraping completed and VPN disconnected.")

        driver.quit()
