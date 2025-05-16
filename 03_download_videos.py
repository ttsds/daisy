import os
import subprocess
import random
import time
from pathlib import Path
import yt_dlp
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument("--use_vpn", action="store_true")
parser.add_argument(
    "--language_code",
    type=str,
    help="Specific language code to download (e.g., 'en', 'es')",
)
args = parser.parse_args()


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


minute_15 = 900

base_dir = Path("videos")

# Filter language dictionary if language_code is specified
if args.language_code:
    if args.language_code not in language_dict:
        print(
            f"Error: Language code '{args.language_code}' not found in supported languages."
        )
        print(f"Supported languages: {', '.join(language_dict.keys())}")
        exit(1)
    language_codes = [args.language_code]
else:
    language_codes = language_dict.keys()

for lang_code in language_codes:
    city, country, _ = language_dict[lang_code]
    lang_folder = base_dir / lang_code
    urls_file = lang_folder / "video_urls.csv"

    if not urls_file.exists():
        print(f"\n⚠️ Skipping '{lang_code}' - no URL file found.")
        continue

    # Connect VPN aligned with language location
    # windscribe_location = f"{city}"
    # windscribe_connect(windscribe_location)
    if args.use_vpn:
        windscribe_location = f"{city}"
        windscribe_connect(windscribe_location)

    # Load URLs
    with urls_file.open("r") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Ensure download directory exists
    download_dir = base_dir / lang_code / "downloaded"
    download_dir.mkdir(exist_ok=True)

    print(f"\n📂 Processing '{lang_code}' videos ({len(urls)} URLs)...")

    # Download videos
    for idx, url in enumerate(urls, 1):
        url_id = url.split(",")[0]
        url_title = url.split(",")[1]
        skipped = False
        try:
            had_error = False
            if Path(f"{download_dir}/{url_id}.mp3").exists():
                print(f"\n➡️ [{lang_code}] Skipping ({idx}/{len(urls)}): {url_title}")
                skipped = True
                continue
            print(f"\n➡️ [{lang_code}] Downloading ({idx}/{len(urls)}): {url_title}")
            subprocess.run(
                [
                    "yt-dlp",
                    "-f",
                    "(bestaudio)[protocol!*=dash]",
                    "--external-downloader",
                    "ffmpeg",
                    "--external-downloader-args",
                    f"ffmpeg_i:-ss {minute_15} -t {60*5}",
                    "--output",
                    f"{download_dir}/{url_id}.%(ext)s",
                    "--extract-audio",
                    "--audio-format",
                    "mp3",
                    "--audio-quality",
                    "64K",
                    "--limit-rate",
                    "500K",
                    "--retries",
                    "10",
                    "--fragment-retries",
                    "10",
                    "--skip-unavailable-fragments",
                    "--user-agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "--quiet",
                    url_id,
                    "--cookies",
                    "cookies.txt",
                ],
                check=True,
            )
            print(f"✅ [{lang_code}] Successfully downloaded.")
        except Exception as e:
            print(f"❌ [{lang_code}] Error: {e}")
            had_error = True
        finally:
            if not skipped and not had_error:
                sleep_duration = random.randint(30, 60)
                print(f"🕒 Sleeping for {sleep_duration}s...")
                time.sleep(sleep_duration)

    print(f"\n🎉 Finished '{lang_code}' folder.")

# Disconnect VPN after completion
if args.use_vpn:
    subprocess.run(["windscribe", "disconnect"], check=True)
    print("\n🚩 All downloads completed and VPN disconnected.")
