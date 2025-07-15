# YouTube Video Scraper

This script uses Selenium to scrape YouTube search results instead of the YouTube API. It searches for videos across multiple languages and categories.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install Chrome browser (if not already installed)

3. Install ChromeDriver:
```bash
# On Ubuntu/Debian
sudo apt-get install chromium-chromedriver

# Or download from: https://chromedriver.chromium.org/
```

4. Set up environment variables (optional):
```bash
# Create a .env file
echo "GEONAMES_USERNAME=your_geonames_username" > .env
```

5. Install Windscribe VPN (optional, for location-based scraping):
```bash
# Download and install Windscribe from: https://windscribe.com/
# Then install the CLI tool
```

## Usage

Run the script to scrape videos for all languages:
```bash
python 01_get_videos.py
```

Or specify a specific language:
```bash
python 01_get_videos.py --language_code en
```

To use Windscribe VPN for location-based scraping (recommended for better results):
```bash
python 01_get_videos.py --use_vpn
python 01_get_videos.py --language_code en --use_vpn
```

## Features

- **Multi-language support**: Searches for videos in 15 different languages
- **Multiple search terms**: Uses 20 different search terms per language
- **Selenium-based scraping**: No API keys required
- **VPN integration**: Optional Windscribe VPN support for location-based scraping
- **Respectful scraping**: Includes delays between requests
- **Error handling**: Graceful handling of timeouts and errors
- **Resume capability**: Skips already processed search terms

## Output

The script creates JSON files in the `videos/{language}/` directory structure, with each file containing search results for a specific search term in that language.

## Notes

- The script runs Chrome in visible mode (not headless) so you can see the scraping process
- It includes a 2-second delay between requests to be respectful to YouTube's servers
- Video information is extracted from the search results page
- Some metadata (like exact publish dates) may not be available from search results
