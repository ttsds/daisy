from pydantic import BaseModel
from dotenv import load_dotenv
import os
import argparse
import glob
import tqdm
from subprocess import run
from pathlib import Path
import json
from transformers import pipeline
import fasttext

language_detector = fasttext.load_model("lid.176.ftz")

classifier = pipeline(
    "zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", type=str, required=True)
parser.add_argument("--language_code", type=str, required=True)
args = parser.parse_args()


def get_utterances(audio_path):

    srt_file = Path(audio_path).with_suffix(".srt")

    if not srt_file.exists():
        try:
            if args.language_code in ["zh", "ja", "ar"]:
                run(
                    [
                        "python",
                        "whisper-diarization/diarize.py",
                        "-a",
                        audio_path,
                        "--whisper-model",
                        "turbo",
                        "--temp-dir",
                        f"temp_outputs_{args.language_code}",
                        "--language",
                        args.language_code,
                    ],
                    check=True,
                )
            else:
                run(
                    [
                        "python",
                        "whisper-diarization/diarize.py",
                        "-a",
                        audio_path,
                        "--whisper-model",
                        "turbo",
                        "--temp-dir",
                        f"temp_outputs_{args.language_code}",
                    ],
                    check=True,
                )
        except Exception as e:
            print(f"Error diarizing {audio_path}: {e}")
            return {}

    with open(srt_file, "r") as f:
        srt_content = f.read()

    # 10 subtitles per prompt
    subtitles = srt_content.split("\n\n")
    # remove empty subtitles
    subtitles = [subtitle for subtitle in subtitles if subtitle.strip()]

    # get unique speakers
    speakers = []
    for subtitle in subtitles:
        speaker = subtitle.split("\n")[2].split(":")[0]
        if speaker not in speakers:
            speakers.append(speaker)

    # create a dictionary of speakers with their sentences
    speakers_dict = {speaker: [] for speaker in speakers}
    for subtitle in subtitles:
        speaker = subtitle.split("\n")[2].split(":")[0]
        text = subtitle.split("\n")[2].split(":")[1:]
        if isinstance(text, list):
            text = ":".join(text)
        text = text.strip()
        # we filter length differently for different languages
        # Language	ISO Code	Min Characters	Max Characters	Rationale
        # English	en	10	250	Moderate phoneme density; moderate sentence length
        # Spanish	es	10	260	Similar density to English; slightly longer words/sentences
        # Italian	it	10	260	Similar density to Spanish; expressive sentences
        # Japanese	ja	5	150	Compact due to logograms/kana; higher density per character
        # Polish	pl	12	270	Longer average words and consonant clusters
        # Portuguese	pt	10	260	Similar phonological density to Spanish
        # Turkish	tr	10	230	Agglutinative; longer average words but concise sentence structure
        # Chinese	zh	5	120	Extremely dense logographic script; shorter average sentences
        # French	fr	10	270	Similar to English but slightly longer due to complex morphology
        # German	de	12	300	Longer compound words; complex sentence structures
        # Korean	ko	5	180	Dense hangul syllables; shorter than alphabetic scripts
        # Arabic	ar	10	250	Rich morphology, average density similar to Romance languages
        # Russian	ru	12	270	Moderate complexity; slightly longer word structure
        # Dutch	nl	10	260	Similar to English; moderate sentence length
        # Hindi	hi	10	220	Dense Devanagari script; moderately compact
        if args.language_code == "en":
            min_char = 10
            max_char = 250
        elif args.language_code == "es":
            min_char = 10
            max_char = 260
        elif args.language_code == "it":
            min_char = 10
            max_char = 260
        elif args.language_code == "ja":
            min_char = 5
            max_char = 150
        elif args.language_code == "pl":
            min_char = 12
            max_char = 270
        elif args.language_code == "pt":
            min_char = 10
            max_char = 260
        elif args.language_code == "tr":
            min_char = 10
            max_char = 230
        elif args.language_code == "zh":
            min_char = 5
            max_char = 120
        elif args.language_code == "fr":
            min_char = 10
            max_char = 270
        elif args.language_code == "de":
            min_char = 12
            max_char = 300
        elif args.language_code == "ko":
            min_char = 5
            max_char = 180
        elif args.language_code == "ar":
            min_char = 10
            max_char = 250
        elif args.language_code == "ru":
            min_char = 12
            max_char = 270
        elif args.language_code == "nl":
            min_char = 10
            max_char = 260
        elif args.language_code == "hi":
            min_char = 10
            max_char = 220
        min_char = int(min_char * 1.5)
        if len(text) < min_char:
            continue
        if len(text) > max_char:
            continue
        # check if any controversial topics are mentioned
        candidate_labels = "negative,political,gender,religion,sexual,controversial,rare word,incomplete,race".split(
            ","
        )
        if args.language_code == "en":
            candidate_thresholds = [0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.6, 0.9, 0.6]
        else:
            candidate_thresholds = [
                0.95,
                0.95,
                0.95,
                0.95,
                0.95,
                0.95,
                0.95,
                0.95,
                0.95,
            ]
        result = classifier(text, candidate_labels, multi_label=True)
        topics = [
            label
            for i, (label, score) in enumerate(zip(result["labels"], result["scores"]))
            if score > candidate_thresholds[i]
        ]
        if len(topics) > 0:
            print(f"Skipping {text} because it contains controversial topics: {topics}")
            continue
        # check if its in the target language
        pred = language_detector.predict(text)[0][0].split("__")[-1]
        if pred != args.language_code:
            print(f"Skipping {text} because it is not in {args.language_code}")
            continue
        timestamp = subtitle.split("\n")[1]
        speakers_dict[speaker].append(
            {
                "text": text,
                "timestamp": timestamp,
            }
        )

    # remove speakers with less than 2 sentences
    speakers_dict = {k: v for k, v in speakers_dict.items() if len(v) >= 3}

    return speakers_dict


if __name__ == "__main__":
    files = sorted(list(glob.glob(os.path.join(args.audio_path, "*.mp3"))))
    for file in tqdm.tqdm(files):
        json_file = Path(file).with_suffix(".json")
        if json_file.exists() and not open(json_file, "r").read().strip() == "{}":
            print(f"Skipping {file} because it already exists")
            continue
        if file.endswith(".mp3"):
            print(f"Processing {file}")
            result = get_utterances(file)
            with open(json_file, "w") as f:
                json.dump(result, f, indent=4)
                print(f"Saved {json_file}")
