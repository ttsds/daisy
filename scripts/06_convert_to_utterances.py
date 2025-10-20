import os
import json
import wget
from typing import Tuple
from tempfile import TemporaryDirectory
import subprocess
import glob
import random
from typing import List
import torch
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console

from daisy.processing.utterances import SpeakerExtractor
from daisy.core import LANGUAGES


load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
console = Console()


if __name__ == "__main__":
    # Calculate total number of samples across all languages
    total_samples = 0
    language_sample_counts = {}
    
    for language in LANGUAGES.values():
        sample_path = f"../daisy_dataset/{language.iso2}/samples"
        sample_paths = glob.glob(os.path.join(sample_path, "*.wav"))
        language_sample_counts[language.iso2] = len(sample_paths)
        total_samples += len(sample_paths)
    
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
        
        for language in LANGUAGES.values():
            console.print(f"\n[bold blue]Processing {language.english_name} ({language.native_name})[/bold blue]")
            
            speaker_extractor = SpeakerExtractor(
                language=language.iso2, 
                output_dir=f"../daisy_dataset/{language.iso2}/utterances", 
                device=device
            )
            
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

            del speaker_extractor
            torch.cuda.empty_cache()
    
    console.print("\n[bold green]All processing completed![/bold green]")