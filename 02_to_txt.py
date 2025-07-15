from pathlib import Path
import json
import fasttext
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser(
    description="Process videos in specific language directory"
)
parser.add_argument(
    "--language_code",
    type=str,
    help="Language code to process (if not specified, process all languages)",
)
args = parser.parse_args()

language_detector = fasttext.load_model("lid.176.ftz")

# each directory is a language
video_dirs = (
    [Path("videos") / args.language_code]
    if args.language_code
    else Path("videos").iterdir()
)

for language in video_dirs:
    # Skip if directory doesn't exist
    if not language.exists():
        print(f"Directory for language '{language.name}' not found")
        continue

    # create a txt file with only the video urls in each language directory
    language_code = language.name
    skip_count = 0
    skip_langs = []
    with open(language / "video_urls.csv", "w", encoding="utf-8") as url_file:
        for video in language.glob("*.json"):
            with open(video, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data["items"]:
                    if "video" not in item["id"]["kind"]:
                        continue
                    if language_code == "hi":
                        # hindi has a special case, it can be in hindi or english
                        url_file.write(
                            f"{item['id']['videoId']},{item['snippet']['title']}\n"
                        )
                    # check if title and description is in the target language
                    test_text_title = item["snippet"]["title"].replace("\n", " ")
                    test_text_description = item["snippet"]["description"].replace(
                        "\n", " "
                    )
                    pred_title = language_detector.predict(test_text_title)[0][0].split(
                        "__"
                    )[-1]
                    pred_description = language_detector.predict(test_text_description)[
                        0
                    ][0].split("__")[-1]
                    if (
                        pred_title != language_code
                        and pred_description != language_code
                    ):
                        skip_count += 1
                        skip_langs.append(f"{pred_title} {pred_description}")
                    url_file.write(
                        f"{item['id']['videoId']},{item['snippet']['title']}\n"
                    )
    print(f"Skipped {skip_count} videos in {language_code}")
    print(f"Skipped languages: {skip_langs}")
