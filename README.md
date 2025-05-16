# TTSDS: Multilingual Text-to-Speech Dataset Pipeline

A robust data collection and processing pipeline for creating high-quality multilingual speech datasets for TTS benchmarking.

## Overview

This pipeline automates the collection, processing, and filtering of multilingual speech data from online sources to create a clean, standardized dataset suitable for training and evaluating Text-to-Speech (TTS) systems across multiple languages.

## Supported Languages

The pipeline currently supports 15 languages:
- English (en)
- Spanish (es)
- Italian (it)
- Japanese (ja)
- Polish (pl)
- Portuguese (pt)
- Turkish (tr)
- Chinese (zh)
- French (fr)
- German (de)
- Korean (ko)
- Arabic (ar)
- Russian (ru)
- Dutch (nl)
- Hindi (hi)

## Pipeline Stages

The data processing pipeline consists of the following stages:

1. **Video Discovery** (`01_get_videos.py`): Searches for relevant videos in target languages using YouTube API with geo-specific queries.

2. **Language Verification** (`02_to_txt.py`): Extracts video metadata and verifies that content is in the target language using fastText language detection.

3. **Video Download** (`03_download_videos.py`): Downloads the verified videos for further processing.

4. **Utterance Extraction** (`04_get_utterances.py`): Processes videos to extract individual utterances and associated metadata.

5. **Audio Processing** (`05_extract_and_filter.py`): Extracts audio segments, performs quality filtering based on:
   - Speech quality (DNSMOS score)
   - Background noise/music detection
   - Transcript accuracy (WER/CER metrics)
   - Audio duration constraints

6. **Final Dataset Creation** (`06_final_filter.py`): Performs final filtering and creates the standardized dataset.

## Requirements

- Python 3.8+
- YouTube Data API key
- GeoNames API username
- GPU recommended for audio processing (with CUDA support)

### Python Dependencies
- torch/torchaudio
- faster_whisper
- fasttext
- demucs
- transformers
- rich
- requests
- geopy
- python-dotenv

## Usage

1. **Setup Environment**:
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/TTSDS.git
   cd TTSDS
   
   # Create and activate a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up API keys in .env file
   echo "YOUTUBE_API_KEYS=your_youtube_api_key" > .env
   echo "GEONAMES_USERNAME=your_geonames_username" >> .env
   ```

2. **Run the Pipeline**:

   To process a specific language:
   ```bash
   python 01_get_videos.py --language_code en
   python 02_to_txt.py --language_code en
   python 03_download_videos.py --language_code en
   python 04_get_utterances.py --language_code en
   python 05_extract_and_filter.py --language_code en --device_index 0
   python 06_final_filter.py --language_code en
   ```

   To process all supported languages (may take significant time):
   ```bash
   for lang in en es it ja pl pt tr zh fr de ko ar ru nl hi; do
       python 01_get_videos.py --language_code $lang
       python 02_to_txt.py --language_code $lang
       python 03_download_videos.py --language_code $lang
       python 04_get_utterances.py --language_code $lang
       python 05_extract_and_filter.py --language_code $lang --device_index 0
       python 06_final_filter.py --language_code $lang
   done
   ```

## Dataset Output Format

The final dataset will be organized by language code with the following structure:
```
dataset/
├── en/
│   ├── 0001_speaker1_0.mp3
│   ├── 0001_speaker1_0.txt
│   ├── ...
├── es/
│   ├── ...
```

Each audio file has a corresponding text file containing the transcript. The naming convention is:
`{index}_{speaker_hash}_{utterance_id}.{ext}`

## Quality Metrics

The pipeline filters utterances based on:
- Word Error Rate (WER < 10%)
- Character Error Rate (CER < 5%)
- Duration (3-30 seconds)
- Background music/noise level

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses the YouTube Data API for video discovery
- Audio processing leverages various open-source tools including faster-whisper, demucs, and torchaudio
