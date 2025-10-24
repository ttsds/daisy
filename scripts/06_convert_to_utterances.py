import os
import json
import wget
from typing import Tuple
from tempfile import TemporaryDirectory
import subprocess
import glob
import random
from typing import List
import argparse

import torch
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console

from daisy.processing.utterances import SpeakerExtractor
from daisy.core import LANGUAGES
from daisy.asr import MMSASRModel

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
console = Console()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language_range", type=str, default=None)
    args = parser.parse_args()

    lang_vals = sorted(LANGUAGES.values(), key=lambda x: x.iso2)

    if args.language_range is not None:
        start_idx, end_idx = args.language_range.split(":")
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        lang_vals = lang_vals[start_idx:end_idx]

    # try languages from mms
    mms = MMSASRModel(device=device, language="en")

    for language in lang_vals:
        mms.set_language(language.iso2)


    # Calculate total number of samples across all languages
    total_samples = 0
    language_sample_counts = {}
    
    for language in lang_vals:
        sample_path = f"../daisy_dataset/{language.iso2}/samples"
        sample_paths = glob.glob(os.path.join(sample_path, "*.wav"))
        language_sample_counts[language.iso2] = len(sample_paths)
        total_samples += len(sample_paths)
    
    speaker_extractor = SpeakerExtractor(
        language=language.iso2, 
        output_dir=f"../daisy_dataset/{language.iso2}/utterances", 
        device=device
    )

    # Create progress bar for overall processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True
    ) as progress:
        overall_task = progress.add_task("Processing all languages", total=total_samples)
        
        for language in lang_vals:
            if os.path.exists(f"../daisy_dataset/{language.iso2}/utterances"):
                dirs = os.listdir(f"../daisy_dataset/{language.iso2}/utterances")
                wav_files = glob.glob(os.path.join(f"../daisy_dataset/{language.iso2}/samples", "*.wav"))
                len_tolerance = 10
                if abs(len(wav_files) - len(dirs)) <= len_tolerance:
                    print(f"Skipping {language.english_name} ({language.native_name}) because it already has utterances (total: {len(dirs)}, wav files: {len(wav_files)})")
                    progress.advance(overall_task, len(wav_files))
                    continue

            console.print(f"\n[bold blue]Processing {language.english_name} ({language.native_name})[/bold blue]")
            

            speaker_extractor.set_language(language.iso2)
            speaker_extractor.set_output_dir(f"../daisy_dataset/{language.iso2}/utterances")

            sample_path = f"../daisy_dataset/{language.iso2}/samples"
            sample_paths = glob.glob(os.path.join(sample_path, "*.wav"))
            
            # Create progress bar for current language
            language_task = progress.add_task(
                f"Processing {language.english_name} samples", 
                total=len(sample_paths)
            )
            
            for sample_path in sample_paths:
                speaker_extractor.extract(sample_path)
                progress.advance(language_task)
                progress.advance(overall_task)
            
            progress.remove_task(language_task)
    
    console.print("\n[bold green]All processing completed![/bold green]")