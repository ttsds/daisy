from pathlib import Path
import json
import argparse
import torch
from faster_whisper import WhisperModel
import torchaudio
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.functional.text import word_error_rate, char_error_rate
from hashlib import md5
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import os
import re
from transformers import AutoModel
import sys
import demucs.separate
import tempfile

# Removed voicerestore imports and sys.path modifications

# faster_whisper_model = "distil-large-v3"
faster_whisper_model = "turbo"

SAMPLING_RATE = 16000

parser = argparse.ArgumentParser()
# parser.add_argument("--input_dir", type=str, required=True)
# parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--language_code", type=str, required=True)
parser.add_argument("--device_index", type=int, required=False, default=0)
args = parser.parse_args()

args.input_dir = f"videos/{args.language_code}/downloaded"
args.output_dir = f"videos/{args.language_code}/processed"

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)

# remove all files in output_dir
for file in output_dir.glob("*.mp3"):
    file.unlink()
for file in output_dir.glob("*.txt"):
    file.unlink()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = "float32"
whisper = WhisperModel(
    faster_whisper_model,
    device=device,
    device_index=args.device_index,
    compute_type="float32",
)
dnsmos = DeepNoiseSuppressionMeanOpinionScore(SAMPLING_RATE, False)

# Removed BigVGAN and OptimizedAudioRestorationModel initialization


def get_background_music_energy(audio):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        audio_file = temp_dir / "audio.mp3"
        torchaudio.save(audio_file, audio, SAMPLING_RATE)
        demucs.separate.main(
            [
                "--mp3",
                "--two-stems",
                "vocals",
                "-o",
                str(temp_dir),
                "-n",
                "htdemucs",
                str(audio_file),
            ]
        )
        vocals_file = temp_dir / "htdemucs" / "audio" / "vocals.mp3"
        other_file = temp_dir / "htdemucs" / "audio" / "no_vocals.mp3"

        vocals_audio, _ = torchaudio.load(vocals_file)
        other_audio, _ = torchaudio.load(other_file)

        print(f"Vocals energy: {torch.sum(vocals_audio[0] ** 2)}")
        print(f"Other energy: {torch.sum(other_audio[0] ** 2)}")

        return torch.sum(other_audio[0] ** 2)


def timestamp_to_samples(timestamp):
    timestamp_seconds = (
        int(timestamp.split(":")[0]) * 3600
        + int(timestamp.split(":")[1]) * 60
        + int(timestamp.split(":")[2].split(",")[0])
        + float(timestamp.split(":")[2].split(",")[1]) / 1000
    )
    return int(timestamp_seconds * SAMPLING_RATE)


def normalize_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).strip().lower()


# Removed process_with_voicerestore function

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize rich console
console = Console()

files = list(Path(args.input_dir).glob("*.json"))
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    main_task = progress.add_task("[blue]Processing files...", total=len(files))
    index = 0
    stop_processing = False
    for file_path in files:
        if stop_processing:
            break
        with open(file_path, "r") as f:
            data = json.load(f)
            audio_file = file_path.with_suffix(".mp3")
            progress.update(
                main_task, description=f"[blue]Processing: {audio_file.name}"
            )

            audio, sr = torchaudio.load(audio_file)
            if sr != SAMPLING_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLING_RATE)

            for speaker in data.keys():
                speaker_clean = False
                speaker_num_saved = 0
                speaker_hash = md5(f"{file_path.stem}_{speaker}".encode()).hexdigest()
                console.print(
                    f"\n[bold green]Speaker:[/] {speaker} ([dim]{speaker_hash}[/dim])"
                )

                for i, utterance in enumerate(data[speaker]):
                    if speaker_num_saved >= 8:
                        continue
                    if stop_processing:
                        break
                    text = utterance["text"]
                    timestamp = utterance["timestamp"]
                    # example timestamp: "00:00:59,640 --> 00:01:09,220"
                    timestamp_start = timestamp.split("-->")[0].strip()
                    timestamp_end = timestamp.split("-->")[1].strip()
                    timestamp_start_samples = timestamp_to_samples(timestamp_start)
                    timestamp_end_samples = timestamp_to_samples(timestamp_end)
                    utterance_audio = audio[
                        :, timestamp_start_samples:timestamp_end_samples
                    ]
                    if utterance_audio.dim() > 1:
                        utterance_audio = torch.mean(
                            utterance_audio, dim=0, keepdim=True
                        )
                    audio_duration = (
                        timestamp_end_samples - timestamp_start_samples
                    ) / SAMPLING_RATE
                    output_file = output_dir / f"{index:04d}_{speaker_hash}_{i}.mp3"

                    # Calculate MOS on original audio only
                    # mos = dnsmos(utterance_audio)[-1]
                    mos = 1
                    # Save original audio to temp file for whisper
                    torchaudio.save(output_file, utterance_audio, SAMPLING_RATE)

                    segments, _ = whisper.transcribe(output_file, beam_size=5)
                    transcript = " ".join([segment.text for segment in segments])
                    # replace double spaces with single spaces
                    transcript = re.sub(r"\s+", " ", transcript)
                    text = re.sub(r"\s+", " ", text)
                    # remove leading and trailing spaces
                    transcript = transcript.strip()
                    text = text.strip()

                    transcript_normalized = normalize_text(transcript)
                    text_normalized = normalize_text(text)

                    # Create a table for the utterance results
                    table = Table(title=f"Utterance {i}")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="magenta")

                    if mos < 2.5:
                        mos_prefix = "[red]"
                    else:
                        mos_prefix = "[green]"
                        speaker_clean = True

                    if speaker_clean:
                        mos = float("inf")

                    table.add_row("Original Text", text)
                    table.add_row("Transcript", transcript)
                    table.add_row("MOS Score", f"{mos_prefix}{mos:.2f}")

                    wer = word_error_rate(transcript_normalized, text_normalized)
                    if wer < 0.1:
                        wer_prefix = "[green]"
                    else:
                        wer_prefix = "[red]"
                    table.add_row(
                        "Word Error Rate",
                        f"{wer_prefix}{wer:.2%}[/]",
                    )

                    cer = char_error_rate(transcript_normalized, text_normalized)
                    if cer < 0.05:
                        cer_prefix = "[green]"
                    else:
                        cer_prefix = "[red]"
                    table.add_row(
                        "Character Error Rate",
                        f"{cer_prefix}{cer:.2%}[/]",
                    )

                    console.print(Panel(table, border_style="blue"))
                    console.print("---")

                    background_music_energy = get_background_music_energy(
                        utterance_audio
                    )

                    if np.isnan(wer) or np.isnan(cer):
                        wer = 0
                        cer = 0

                    if (
                        mos < 0
                        or wer > 0.1
                        or cer > 0.05
                        or audio_duration < 3
                        or audio_duration > 30
                        or background_music_energy > 10
                    ):
                        console.print(f"[red]Skipping utterance {i}[/red]")
                        if wer > 0.1 or cer > 0.1:
                            console.print(
                                f"[red]Word Error Rate: {wer:.2%}[/red] [red]Character Error Rate: {cer:.2%}[/red]"
                            )
                        elif audio_duration < 3:
                            console.print(
                                f"[red]Audio duration too short: {audio_duration:.2f} seconds[/red]"
                            )
                        elif audio_duration > 30:
                            console.print(
                                f"[red]Audio duration too long: {audio_duration:.2f} seconds[/red]"
                            )
                        elif background_music_energy > 10:
                            console.print(
                                f"[red]Background music energy too high: {background_music_energy:.2f}[/red]"
                            )
                        # Clean up temporary files
                        os.remove(output_file)
                    else:
                        console.print(f"[green]Saving utterance {index}[/green]")
                        index += 1
                        speaker_num_saved += 1
                        with open(output_file.with_suffix(".txt"), "w") as f:
                            f.write(text)
                        if index >= 200:
                            stop_processing = True
                            break
                if speaker_num_saved == 1:
                    # delete all files of this speaker
                    for file in output_dir.glob(f"*_{speaker_hash}_*.mp3"):
                        os.remove(file)
                    for file in output_dir.glob(f"*_{speaker_hash}_*.txt"):
                        os.remove(file)
                    index -= 1

        progress.advance(main_task)
