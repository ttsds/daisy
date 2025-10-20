"""
Speaker diarization using NeMo models.
"""

# from: https://github.com/MahmoudAshraf97/whisper-diarization/blob/fcbd1930d8a2fb2dc4e7cd0b7a0f2bffb786e8d3/diarization/msdd/msdd.py
import json
import os
import tempfile
from typing import Union

import torch
import torchaudio
from nemo.collections.asr.models import ClusteringDiarizer as ClusteringModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer as MSDDModel
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
from omegaconf import OmegaConf
import numpy as np


class NeuralDiarizer:
    def __init__(self, device: Union[str, torch.device]):
        self.model_config = self._create_config()
        self.model: MSDDModel = MSDDModel(cfg=self.model_config).to(device)

    def _create_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "diarization", "diar_infer_telephonic.yaml"
        )
        config = OmegaConf.load(config_path)

        # Restore the necessary configuration overrides
        config.diarizer.manifest_filepath = None
        config.diarizer.out_dir = (
            None  # Must be set to None to avoid writing to current directory
        )

        # Paths to pretrained models
        pretrained_vad = "vad_multilingual_marblenet"
        pretrained_speaker_model = "titanet_large"
        pretrained_msdd = "diar_msdd_telephonic"

        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.msdd_model.model_path = pretrained_msdd
        config.diarizer.vad.model_path = pretrained_vad

        # These are critical for inference
        config.diarizer.oracle_vad = False
        config.diarizer.clustering.parameters.oracle_num_speakers = False

        return config

    def diarize(self, sr: int, audio: np.ndarray):
        audio = torch.from_numpy(audio).unsqueeze(0)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)

        with tempfile.TemporaryDirectory() as temp_path:
            audio_path = os.path.join(temp_path, "mono_file.wav")
            torchaudio.save(audio_path, audio, 16000, channels_first=True)

            manifest_path = os.path.join(temp_path, "manifest.json")
            meta = {
                "audio_filepath": audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "rttm_filepath": None,
                "uem_filepath": None,
            }
            with open(manifest_path, "w") as f:
                json.dump(meta, f)

            self.model._initialize_configs(
                manifest_path=manifest_path,
                max_speakers=8,
                num_speakers=None,
                tmpdir=temp_path,
                batch_size=24,
                num_workers=0,
                verbose=True,
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.out_dir = (
                temp_path
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.manifest_filepath = (
                manifest_path
            )
            self.model.msdd_model.cfg.test_ds.manifest_filepath = manifest_path
            self.model.diarize()

            pred_rttm_path = os.path.join(temp_path, "pred_rttms", "mono_file.rttm")
            pred_labels = rttm_to_labels(pred_rttm_path)

            labels = []
            for label in pred_labels:
                start, end, speaker = label.split()
                labels.append((float(start), float(end), int(speaker.split("_")[1])))

            return sorted(labels, key=lambda x: x[0])


class ClusteringDiarizer:
    def __init__(self, device: Union[str, torch.device]):
        self.model_config = self._create_config()
        self.model: ClusteringModel = ClusteringModel(cfg=self.model_config).to(device)

    def _create_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "diarization", "diar_infer_meeting.yaml")
        config = OmegaConf.load(config_path)

        # Restore the necessary configuration overrides
        config.diarizer.manifest_filepath = None
        config.diarizer.out_dir = None

        # Paths to pretrained models
        pretrained_vad = "vad_multilingual_marblenet"
        pretrained_speaker_model = "titanet_large"

        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.vad.model_path = pretrained_vad

        config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [
            1.5,
            1.25,
            1.0,
            0.75,
            0.5,
        ]
        config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [
            0.75,
            0.625,
            0.5,
            0.375,
            0.1,
        ]
        config.diarizer.speaker_embeddings.parameters.multiscale_weights = [
            1,
            1,
            1,
            1,
            1,
        ]

        # These are critical for inference
        config.diarizer.oracle_vad = False
        config.diarizer.clustering.parameters.oracle_num_speakers = False

        return config

    def diarize(self, sr: int, audio: np.ndarray):
        audio = torch.from_numpy(audio).unsqueeze(0)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)

        with tempfile.TemporaryDirectory() as temp_path:
            audio_path = os.path.join(temp_path, "mono_file.wav")
            torchaudio.save(audio_path, audio, 16000, channels_first=True)

            manifest_path = os.path.join(temp_path, "manifest.json")
            meta = {
                "audio_filepath": audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "rttm_filepath": None,
                "uem_filepath": None,
            }
            with open(manifest_path, "w") as f:
                json.dump(meta, f)

            # Set the manifest and output directory at runtime

            print(dir(self.model))
            self.model._cfg.diarizer.manifest_filepath = manifest_path
            self.model._cfg.diarizer.out_dir = temp_path
            self.model.diarize()

            pred_rttm_path = os.path.join(temp_path, "pred_rttms", "mono_file.rttm")
            pred_labels = rttm_to_labels(pred_rttm_path)

            labels = []
            for label in pred_labels:
                start, end, speaker = label.split()
                labels.append((float(start), float(end), int(speaker.split("_")[1])))

            return sorted(labels, key=lambda x: x[0])
