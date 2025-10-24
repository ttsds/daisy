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
from hashlib import sha256
import random

import numpy as np
import torch
import torchaudio
from sklearn.metrics.pairwise import cosine_similarity
from nemo.collections.asr.models import (
    EncDecSpeakerLabelModel,
)
from humanhash import humanize
from openai import OpenAI
import pandas as pd

from . import NeuralDiarizer, ClusteringDiarizer
from daisy.utils import find_valleys, wada_snr, DemucsProcessor
from daisy.utils.retry import llm_api_retry
from daisy.core.models import Utterance, UtteranceList, Speaker, SimpleUtterance
from daisy.asr import WhisperASRModel, MMSASRModel
from daisy.core import LANGUAGES



class SpeakerExtractor():
    """
    Given an audio sample, this diarizes it, groups the speakers into clusters, and extracts the speakers.
    Then it filters by length, SNR, and similarity to the mean speaker embedding.
    Finally, it performs ASR on the filtered speakers and uses the transcripts to pick two utterances per speaker.
    In some cases, a speaker may be dropped completely if there are no utterances that meet the quality thresholds,
    or all the ASR transcripts contain controversial content.
    """
    def __init__(self, language: str, output_dir: str, device: str = "cpu", overwrite: bool = False, **kwargs):
        self.device = device
        self.output_dir = output_dir
        self.overwrite = overwrite
        default_kwargs = {
            "max_segment_seconds": 25.0,
            "valley_min_gap_seconds": 3.0,
            "valley_sigma": 100,
            "speaker_model_id": "nvidia/speakerverification_en_titanet_large",
            "min_similarity": 0.8,
            "min_snr": 15.0,
            "min_duration": 1.5,
            "max_duration": 25.0,
            "remove_singleton_speakers": True,
            "diarizer_type": "neural",
            "demucs_model_name": "htdemucs",
            "filter_llm_model": "x-ai/grok-4-fast",
            "max_consecutive_failures": 3,
        }
        self.kwargs = {**default_kwargs, **kwargs}

        self.language = language
        self.whisper = WhisperASRModel(device=device)
        self.mms = MMSASRModel(device=device, language=language)
        self.demucs = DemucsProcessor(device=device, model_name=self.kwargs["demucs_model_name"])
        if self.kwargs["diarizer_type"] == "neural":
            self.diarizer = NeuralDiarizer(device)
        elif self.kwargs["diarizer_type"] == "clustering":
            self.diarizer = ClusteringDiarizer(device)
        else:
            raise ValueError(f"Invalid diarizer type: {self.kwargs['diarizer_type']}")

        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(self.kwargs["speaker_model_id"])
        self.speaker_model.to(self.device)

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_KEY"),
        )
        self.llm_id = self.kwargs["filter_llm_model"]
        self.consecutive_failures = 0
        self.max_consecutive_failures = self.kwargs["max_consecutive_failures"]

    @llm_api_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _make_llm_request(self, messages: list, response_format=None):
        """
        Make an LLM request with retry logic.
        """
        if response_format:
            return self.client.chat.completions.parse(
                model=self.llm_id,
                messages=messages,
                response_format=response_format,
            ).choices[0].message.parsed
        else:
            return self.client.chat.completions.create(
                model=self.llm_id,
                messages=messages,
            ).choices[0].message.content

    def extract(self, audio_path: str) -> List[Speaker]:
        """
        Extract speakers from audio file.
        """

        os.makedirs(self.output_dir, exist_ok=True)
        output_dir = os.path.join(self.output_dir, os.path.basename(audio_path).split(".")[0])
        if os.path.exists(output_dir):
            return

        os.makedirs(output_dir, exist_ok=True)

        try:
            audio_item_id = audio_path.split("/")[-1].split(".")[0]
            video_info = pd.read_csv(os.path.dirname(audio_path).replace("/samples", "/sampled_items.csv"))
            video_info = video_info[video_info["audio_item_id"] == audio_item_id]
            video_name = video_info["channel_name"].values[0]
            video_title = video_info["title"].values[0]

            
            audio, sr = self.demucs.process(audio_path)

            try:
                diarization = self.diarizer.diarize(sr, audio)
            # index and value error
            except (IndexError, ValueError) as e:
                print(f"Error diarizing audio: {e}")
                return
            
            speaker_segments = self._get_segments(audio, sr, diarization)

            speaker_embeddings = {}
            segment_embeddings = {}
            segment_id = 0
            for speaker, segments in sorted(speaker_segments.items()):
                speaker_embeddings[speaker] = []
                for segment in segments:
                    audio_embedding, _ = self.speaker_model.infer_segment(segment)
                    audio_embedding = audio_embedding.squeeze(0).cpu().numpy()
                    segment_embeddings[segment_id] = audio_embedding
                    segment_id += 1
                    speaker_embeddings[speaker].append(audio_embedding)
                speaker_embeddings[speaker] = np.stack(speaker_embeddings[speaker])
                mean_embedding = speaker_embeddings[speaker].mean(axis=0)
                speaker_embeddings[speaker] = mean_embedding.reshape(1, -1)

            # update the speaker keys to be the humanhash of the mean speaker embedding
            id2embedding = {}
            skip_speakers = []
            for speaker, embedding in speaker_embeddings.items():
                mean_embedding = embedding.mean(axis=0)
                speaker_name = f"{audio_item_id}_{speaker}"
                if os.path.exists(os.path.join(self.output_dir, speaker_name)):
                    skip_speakers.append(speaker)
                    continue
                id2embedding[speaker] = humanize(sha256(speaker_name.encode()).hexdigest()).replace("-", "_")
            speaker_segments = {id2embedding[speaker]: segments for speaker, segments in speaker_segments.items()}
            speaker_embeddings = {id2embedding[speaker]: embeddings for speaker, embeddings in speaker_embeddings.items()}

            segment_attributes = {}
            segment_id = 0
            for speaker, segments in sorted(speaker_segments.items()):
                if speaker in skip_speakers:
                    continue
                for segment in segments:
                    attributes = self._get_attributes(segment, sr, segment_embeddings[segment_id], speaker_embeddings[speaker])
                    segment_attributes[segment_id] = attributes
                    segment_id += 1

            speaker_utterances = {}
            segment_id = 0
            for speaker, segments in sorted(speaker_segments.items()):
                if speaker in skip_speakers:
                    continue
                speaker_utterances[speaker] = []
                for segment in segments:
                    attributes = segment_attributes[segment_id]
                    if (attributes["similarity"] > self.kwargs["min_similarity"] 
                        and attributes["snr"] > self.kwargs["min_snr"] 
                        and attributes["duration"] > self.kwargs["min_duration"] 
                        and attributes["duration"] < self.kwargs["max_duration"]
                    ):
                        whisper_transcript = self.whisper.transcribe(segment, sr)
                        mms_transcript = self.mms.transcribe(segment, sr)
                        speaker_utterances[speaker].append(Utterance(
                            identifier=f"{speaker}_{segment_id}",
                            audio=segment,
                            sr=sr,
                            texts=[whisper_transcript, mms_transcript],
                            similarity=attributes["similarity"],
                            snr=attributes["snr"],
                            duration=attributes["duration"],
                            audio_item_id=audio_path.split("/")[-1].split(".")[0],
                        ))
                    segment_id += 1

            # sample 10 utterances per speaker if there are more than 10, drop speakers with less than 2 utterances
            del_speakers = []
            for speaker, utterances in speaker_utterances.items():
                if speaker in skip_speakers:
                    continue
                if len(utterances) > 10:
                    random.seed(42)
                    utterances = random.sample(utterances, 10)
                elif len(utterances) < 2:
                    del_speakers.append(speaker)
            for speaker in del_speakers:
                del speaker_utterances[speaker]
            
            for speaker in speaker_utterances.keys():
                if speaker in skip_speakers:
                    continue
                linebreak = "\n"
                prompt = f"""
                There will be a list of utterances for a speaker and their transcripts.
                You need to filter the utterances based on the transcripts, and correct the transcripts if necessary.
                Transcripts containing controversial content should be dropped, meaning content that is likely to be offensive or harmful.
                The language is {LANGUAGES[self.language].english_name}, if the majority of a transcript is not in this language, it should be dropped.
                For additional context, the utterances are extracted from a source video, which is by '{video_name}' and the title is '{video_title}'.
                
                The transcripts are:
                {linebreak.join([f"Id: {utt.identifier}; Transcript 1: '{utt.texts[0]}'; Transcript 2: '{utt.texts[1]}'" for utt in speaker_utterances[speaker]])}
                
                Return the two most representative transcripts for the speaker, which should be useful for subjective evaluation (human listening test).

                Here is an example:
                Input:
                Id: kangaroo_flower_mountain_flow_1; Transcript 1: 'this is a test'; Transcript 2: 'This is test.'
                ...
                Output:
                I decided to keep kangaroo_flower_mountain_flow_1 and correct its transcript to 'This is a test.'
                The second transcript I decided to keep is ...

                Remember: You can also decide to not keep any of the transcripts, if none are appropriate.
                """


                response = self._make_llm_request(
                    messages=[{"role": "user", "content": prompt}]
                )
                
                prompt = f"""
                Take the following unstructured response and return a list of SimpleUtterance objects, which have the following fields:
                - identifier: str
                - text: str
                The response is:
                {response}
                If there are no transcripts to keep, return an empty list.
                """

                response = self._make_llm_request(
                    messages=[{"role": "user", "content": prompt}],
                    response_format=UtteranceList
                )
                if len(response.items) == 0:
                    continue
                
                save_utterances = []
                for item in response.items:
                    for utterance in speaker_utterances[speaker]:
                        if utterance.identifier == item.identifier:
                            utterance.text = item.text
                            save_utterances.append(utterance)
                speaker_utterances[speaker] = save_utterances

                os.makedirs(os.path.join(output_dir, speaker), exist_ok=True)
                for utterance in save_utterances:
                    torchaudio.save(os.path.join(output_dir, speaker, f"{utterance.identifier}.wav"), torch.from_numpy(utterance.audio).unsqueeze(0), utterance.sr)
                    with open(os.path.join(output_dir, speaker, f"{utterance.identifier}.txt"), "w", encoding="utf-8") as f:
                        f.write(utterance.text)
                    with open(os.path.join(output_dir, speaker, f"{utterance.identifier}.json"), "w", encoding="utf-8") as f:
                        json_output = {}
                        json_output["identifier"] = utterance.identifier
                        json_output["texts"] = utterance.texts + [utterance.text]
                        json_output["snr"] = utterance.snr
                        json_output["similarity"] = utterance.similarity
                        json_output["duration"] = utterance.duration
                        json_output["audio_item_id"] = audio_item_id
                        json.dump(json_output, f, ensure_ascii=False, indent=2)
            self.consecutive_failures = 0
        except Exception as e:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                raise e
            print(f"Error extracting speakers: {e}")
            return

    def _get_segments(
        self,
        audio: np.ndarray,
        sr: int,
        diarization: List[Tuple[float, float, int]],
    ) -> Dict[int, List[np.ndarray]]:
        """
        Save diarized segments to disk, splitting segments longer than
        max_segment_seconds at valley positions.

        Returns list of saved wav file paths.
        """
        speaker_segments = {}
        for i, (start, end, speaker) in enumerate(diarization):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            audio_start = int(start * sr)
            audio_end = int(end * sr)
            segment = audio[audio_start:audio_end]
            if len(segment) > self.kwargs["max_segment_seconds"] * sr:
                valleys = find_valleys(segment, sr, valley_sigma=self.kwargs["valley_sigma"])
                start_index = 0
                last_valley = 0.0
                split_index = 0
                for j, valley in enumerate(valleys):
                    end_index = int(valley * sr)
                    if valley - last_valley > self.kwargs["valley_min_gap_seconds"]:
                        speaker_segments[speaker].append(segment[start_index:end_index])
                        start_index = end_index
                        last_valley = valley
                        split_index += 1
                    else:
                        continue
                speaker_segments[speaker].append(segment[start_index:])
            elif len(segment) > 0:
                speaker_segments[speaker].append(segment)
        return speaker_segments

    def _get_attributes(self, audio: np.ndarray, sr: int, audio_embedding: np.ndarray, speaker_embedding: np.ndarray) -> Dict[str, float]:
        """
        Get attributes of audio file.
        """
        audio_16khz = torchaudio.transforms.Resample(sr, 16000)(torch.from_numpy(audio).unsqueeze(0)).numpy()[0]
        snr = wada_snr(audio_16khz)
        if len(audio_embedding.shape) == 1:
            audio_embedding = audio_embedding.reshape(1, -1)
        similarity = cosine_similarity(audio_embedding, speaker_embedding).flatten()[0]
        return {
            "duration": len(audio) / sr,
            "snr": snr,
            "similarity": similarity,
        }

    def set_language(self, language: str) -> None:
        self.mms.set_language(language)

    def set_output_dir(self, output_dir: str) -> None:
        self.output_dir = output_dir