"""
High-level pipeline to convert long audio into utterances:
 - Diarize
 - Segment (with valley-based splitting for long segments)
 - Compute per-file stats (duration, similarity, SNR)
 - Filter segments by quality thresholds

This mirrors the behavior of the 06_convert_to_utterances.py script while
making the logic reusable from library code.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch
import torchaudio
from sklearn.metrics.pairwise import cosine_similarity
from nemo.collections.asr.models import (
    EncDecSpeakerLabelModel,
)

from . import NeuralDiarizer, ClusteringDiarizer
from daisy.utils import find_valleys, wada_snr, stem_demucs


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _segment_and_save(
    sr: int,
    audio: np.ndarray,
    diarization: List[Tuple[float, float, int]],
    output_dir: str,
    max_segment_seconds: float = 25.0,
    valley_min_gap_seconds: float = 3.0,
) -> List[str]:
    """
    Save diarized segments to disk, splitting segments longer than
    max_segment_seconds at valley positions.

    Returns list of saved wav file paths.
    """
    saved_paths: List[str] = []
    for i, (start, end, speaker) in enumerate(diarization):
        audio_start = int(start * sr)
        audio_end = int(end * sr)
        segment = audio[audio_start:audio_end]
        if len(segment) > max_segment_seconds * sr:
            valleys = find_valleys(segment, sr)
            start_index = 0
            last_valley = 0.0
            split_index = 0
            for j, valley in enumerate(valleys):
                end_index = int(valley * sr)
                if valley - last_valley > valley_min_gap_seconds:
                    path = os.path.join(
                        output_dir,
                        f"speaker_{speaker}_{i:04d}_{split_index:04d}.wav",
                    )
                    torchaudio.save(
                        path,
                        torch.from_numpy(segment[start_index:end_index]).unsqueeze(0),
                        sr,
                    )
                    saved_paths.append(path)
                    start_index = end_index
                    last_valley = valley
                    split_index += 1
                else:
                    continue
            # save the last chunk
            path = os.path.join(
                output_dir, f"speaker_{speaker}_{i:04d}_{split_index:04d}.wav"
            )
            torchaudio.save(
                path,
                torch.from_numpy(segment[start_index:]).unsqueeze(0),
                sr,
            )
            saved_paths.append(path)
        elif len(segment) > 0:
            path = os.path.join(output_dir, f"speaker_{speaker}_{i:04d}.wav")
            torchaudio.save(
                path,
                torch.from_numpy(segment).unsqueeze(0),
                sr,
            )
            saved_paths.append(path)
    return saved_paths


def _load_or_compute_embeddings(
    wav_files: List[str],
    device: str,
    model_id: str,
) -> Dict[str, Dict]:
    """
    Compute per-file duration and speaker similarity to speaker mean.

    Returns a dict keyed by wav file path with fields: speaker, duration, similarity.
    """
    speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_id)
    speaker_model.to(device)

    speakers = set()
    wav_file_dict: Dict[str, Dict] = {wav_file: {} for wav_file in wav_files}
    for wav_file in wav_files:
        if "stemmed" in wav_file:
            continue
        speaker = wav_file.split("/")[-1].split("_")[1]
        speakers.add(speaker)
        wav_file_dict[wav_file]["speaker"] = speaker
        audio, sr = torchaudio.load(wav_file)
        wav_file_dict[wav_file]["duration"] = float(audio.shape[1] / sr)

    for speaker in speakers:
        speaker_wav_files = [
            wav_file for wav_file in wav_files if f"speaker_{speaker}_" in wav_file
        ]
        embeddings = []
        for wav_file in speaker_wav_files:
            emb = speaker_model.get_embedding(wav_file).squeeze(0).cpu().numpy()
            embeddings.append(emb)
        if len(embeddings) == 0:
            continue
        embeddings = np.stack(embeddings)
        mean_embedding = embeddings.mean(axis=0)
        sims = cosine_similarity(embeddings, mean_embedding.reshape(1, -1)).flatten()
        for i, sim in enumerate(sims):
            wav_file_dict[speaker_wav_files[i]]["similarity"] = float(sim)

    return wav_file_dict


def _add_snr(
    wav_file_dict: Dict[str, Dict]
) -> Dict[str, Dict]:
    """Compute WADA SNR per file and add it to wav_file_dict in-place."""
    for wav_file in list(wav_file_dict.keys()):
        waveform, sr = torchaudio.load(wav_file)
        snr = wada_snr(waveform.numpy()[0])
        wav_file_dict[wav_file]["snr"] = float(snr)
    return wav_file_dict


def _filter_segments(
    wav_file_dict: Dict[str, Dict],
    min_similarity: float = 0.8,
    min_snr: float = 15.0,
    min_duration: float = 1.5,
    max_duration: float = 25.0,
    remove_singleton_speakers: bool = True,
) -> Dict[str, Dict]:
    """
    Apply quality filters and remove speakers that only occur once if requested.
    """
    # threshold filter
    filtered = {
        wav_file: data
        for wav_file, data in wav_file_dict.items()
        if data.get("similarity", 0) > min_similarity
        and data.get("snr", 0) > min_snr
        and data.get("duration", 0) > min_duration
        and data.get("duration", 0) < max_duration
    }

    if not remove_singleton_speakers:
        return filtered

    # remove speakers with only one sample
    speaker_counts: Dict[str, int] = {}
    for _, data in filtered.items():
        spk = data["speaker"]
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
    speakers_to_remove = [spk for spk, count in speaker_counts.items() if count == 1]
    filtered = {
        wav_file: data
        for wav_file, data in filtered.items()
        if data["speaker"] not in speakers_to_remove
    }
    return filtered


def create_utterances(
    audio_path: str,
    output_dir: str,
    device: str = "cpu",
    diarizer_type: str = "neural",
    max_segment_seconds: float = 25.0,
    valley_min_gap_seconds: float = 3.0,
    speaker_model_id: str = "nvidia/speakerverification_en_titanet_large",
    min_similarity: float = 0.8,
    min_snr: float = 15.0,
    min_duration: float = 1.5,
    max_duration: float = 25.0,
    remove_singleton_speakers: bool = True,
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    End-to-end pipeline to produce utterances and quality metadata.

    Returns (all_stats_dict, filtered_stats_dict)
    and writes wav files plus two JSONs to output_dir.
    """
    _ensure_dir(output_dir)

    if os.path.exists(os.path.join(output_dir, "empty.txt")):
        return

    if not os.path.exists(os.path.join(output_dir, "stemmed.wav")):
        sr, audio = stem_demucs(audio_path, device)
        torchaudio.save(
            os.path.join(output_dir, "stemmed.wav"),
            torch.from_numpy(audio).unsqueeze(0),
            sr,
        )
    else:
        audio, sr = torchaudio.load(os.path.join(output_dir, "stemmed.wav"))
        audio = audio.numpy()[0]

    if not os.path.exists(os.path.join(output_dir, "wav_file_dict.json")):
        if diarizer_type == "neural":
            diarizer = NeuralDiarizer(device)
        elif diarizer_type == "clustering":
            diarizer = ClusteringDiarizer(device)
        else:
            raise ValueError(f"Invalid diarizer type: {diarizer_type}")
        diarization = diarizer.diarize(sr, audio)
        _segment_and_save(
            sr,
            audio,
            diarization,
            output_dir,
            max_segment_seconds=max_segment_seconds,
            valley_min_gap_seconds=valley_min_gap_seconds,
        )

    # Collect wavs and compute stats
    if not os.path.exists(os.path.join(output_dir, "wav_file_dict_filtered.json")):
        wav_files = sorted(glob.glob(os.path.join(output_dir, "*.wav")))
        wav_file_dict = _load_or_compute_embeddings(
            wav_files, device=device, model_id=speaker_model_id
        )
        wav_file_dict = _add_snr(wav_file_dict)

        # Save raw stats
        with open(os.path.join(output_dir, "wav_file_dict.json"), "w") as f:
            json.dump(wav_file_dict, f)

        # Filter
        filtered = _filter_segments(
            wav_file_dict,
            min_similarity=min_similarity,
            min_snr=min_snr,
            min_duration=min_duration,
            max_duration=max_duration,
            remove_singleton_speakers=remove_singleton_speakers,
        )

        with open(os.path.join(output_dir, "wav_file_dict_filtered.json"), "w") as f:
            json.dump(filtered, f)

        # remove all files in output_dir except for the ones in filtered
        for file in os.listdir(output_dir):
            if ".json" in file:
                continue
            if "stemmed" in file:
                continue
            if os.path.join(output_dir, file) not in filtered.keys():
                os.remove(os.path.join(output_dir, file))

        if len(os.listdir(output_dir)) == 0:
            with open(os.path.join(output_dir, "empty.txt"), "w") as f:
                f.write("No utterances found")
            return

    return