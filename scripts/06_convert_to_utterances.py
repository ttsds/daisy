import os
import json
import wget
from typing import Tuple
from tempfile import TemporaryDirectory
import subprocess
import glob
import random
from typing import List

import numpy as np
import demucs.separate
import torch
import torchaudio
from nemo.collections.asr.models import (
    EncDecSpeakerLabelModel,
    EncDecClassificationModel,
)
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

from daisy.processing import NeuralDiarizer, ClusteringDiarizer, create_utterances
from daisy.utils import stem_demucs

if __name__ == "__main__":
    paths = glob.glob("../daisy_dataset/de/samples/*.wav")
    # random shuffle the paths
    random.shuffle(paths)
    path_to_stem = paths[0]
    output_path = "test_outputs"
    # Run the reusable pipeline; it will diarize+segment if needed, then compute and
    # filter stats and write the two JSON files to output_path.
    create_utterances(
        audio_path=path_to_stem,
        output_dir=output_path,
        device=device,
    )
