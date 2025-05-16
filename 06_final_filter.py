from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from pathlib import Path
import json
import argparse
import torch
import torchaudio
import shutil
from tqdm import tqdm
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os
import argparse

load_dotenv()

SAMPLING_RATE = 16000


# # device_index argument


def has_crosstalk(audio):
    if not Path(audio).with_suffix(".txt").exists():
        return True
    diarization = pipeline(audio)
    timeline = diarization.get_timeline()
    cross_talk_segments = []
    for segment in timeline.support():
        active_speakers = diarization.crop(segment).labels()
        if len(active_speakers) > 1:
            cross_talk_segments.append((segment.start, segment.end, active_speakers))
    return len(cross_talk_segments) > 0


def main():
    global pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device_index", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline (this may take a minute the first time)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
    )
    pipeline.to(torch.device(f"cuda:{args.device_index}"))

    # Get the list of files in the input directory
    files = list(input_dir.glob("*.mp3"))

    speaker_utterances = {}

    for file in tqdm(files, desc="Grouping files by speaker"):
        speaker = file.stem.split("_")[1]
        if speaker not in speaker_utterances:
            speaker_utterances[speaker] = []
        speaker_utterances[speaker].append(file)

    speaker_utterances_crosstalk = {}
    for speaker, utterances in tqdm(
        speaker_utterances.items(), desc="Computing crosstalk"
    ):
        for utterance in utterances:
            if speaker not in speaker_utterances_crosstalk:
                speaker_utterances_crosstalk[speaker] = []
            crosstalk = has_crosstalk(utterance)
            if crosstalk:
                print(utterance)
            speaker_utterances_crosstalk[speaker].append(crosstalk)

    paired_utterances = []

    while True:
        stop = False
        for speaker, utterances in speaker_utterances.items():
            crosstalk = speaker_utterances_crosstalk[speaker]
            # sort utterances by mos
            zipped = zip(utterances, crosstalk)
            zipped = [(u, c) for u, c in zip(utterances, crosstalk) if not c]
            print(zipped)
            # get the top 2 utterances
            if len(zipped) < 2:
                continue
            top_utterances = zipped[:2]
            paired_utterances.append(
                (
                    top_utterances[0][0],
                    top_utterances[1][0],
                )
            )
            # remove the top 2 utterances from the list
            speaker_utterances[speaker] = [utterance for utterance, _ in zipped[2:]]
            if len(paired_utterances) >= 50:
                stop = True
                break
        if stop:
            break

    # copy the paired utterances and corresponding .txt files to the output directory
    for index, (utterance_1, utterance_2) in tqdm(
        enumerate(paired_utterances), desc="Copying files"
    ):
        utterance_1_audio, sr = torchaudio.load(utterance_1)
        utterance_2_audio, sr = torchaudio.load(utterance_2)
        if sr != SAMPLING_RATE:
            utterance_1_audio = torchaudio.functional.resample(
                utterance_1_audio, sr, SAMPLING_RATE
            )
            utterance_2_audio = torchaudio.functional.resample(
                utterance_2_audio, sr, SAMPLING_RATE
            )
        if not Path(output_dir / "A").exists():
            Path(output_dir / "A").mkdir(parents=True, exist_ok=True)
        if not Path(output_dir / "B").exists():
            Path(output_dir / "B").mkdir(parents=True, exist_ok=True)
        # normalize the audio
        utterance_1_audio = utterance_1_audio / utterance_1_audio.abs().max()
        utterance_2_audio = utterance_2_audio / utterance_2_audio.abs().max()
        torchaudio.save(
            output_dir / "A" / f"{index:04d}.wav",
            utterance_1_audio,
            SAMPLING_RATE,
        )
        torchaudio.save(
            output_dir / "B" / f"{index:04d}.wav",
            utterance_2_audio,
            SAMPLING_RATE,
        )
        shutil.copy(
            utterance_1.with_suffix(".txt"),
            output_dir / "A" / f"{index:04d}.txt",
        )
        shutil.copy(
            utterance_2.with_suffix(".txt"),
            output_dir / "B" / f"{index:04d}.txt",
        )


if __name__ == "__main__":
    main()
