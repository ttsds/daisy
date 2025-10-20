"""
Audio processing utilities for the Daisy pipeline.
"""

import os
import subprocess
from typing import Tuple, List
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torchaudio
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from librosa.feature import melspectrogram
import demucs.api


def find_valleys(audio: np.ndarray, sr: int, valley_sigma: float = 100) -> List[int]:
    """
    Find valleys (silence points) in audio for segmentation.
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        
    Returns:
        List of valley timestamps in seconds
    """
    # resample the audio to 16000 Hz
    audio = torchaudio.transforms.Resample(sr, 16000)(
        torch.from_numpy(audio).unsqueeze(0)
    ).numpy()[0]
    mel_spec = melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec = np.log(mel_spec + 1e-6)
    energy = np.sum(mel_spec, axis=0)
    duration = len(audio) / 16000
    duration_per_frame = duration / mel_spec.shape[1]
    # smooth the energy
    energy = gaussian_filter1d(energy, sigma=valley_sigma)
    valleys = find_peaks(-energy)
    valleys = [valley * duration_per_frame for valley in valleys[0]]
    return valleys


def wada_snr(wav: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio using WADA method.
    
    Direct blind estimation of the SNR of a speech signal.
    Paper: http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    Adapted from: https://labrosa.ee.columbia.edu/projects/snreval/#9
    
    Args:
        wav: Audio waveform as numpy array
        
    Returns:
        SNR value in dB
    """
    # init
    eps = 1e-10
    # next 2 lines define a fancy curve derived from a gamma distribution -- see paper
    db_vals = np.arange(-20, 101)
    g_vals = np.array(
        [
            0.40974774,
            0.40986926,
            0.40998566,
            0.40969089,
            0.40986186,
            0.40999006,
            0.41027138,
            0.41052627,
            0.41101024,
            0.41143264,
            0.41231718,
            0.41337272,
            0.41526426,
            0.4178192,
            0.42077252,
            0.42452799,
            0.42918886,
            0.43510373,
            0.44234195,
            0.45161485,
            0.46221153,
            0.47491647,
            0.48883809,
            0.50509236,
            0.52353709,
            0.54372088,
            0.56532427,
            0.58847532,
            0.61346212,
            0.63954496,
            0.66750818,
            0.69583724,
            0.72454762,
            0.75414799,
            0.78323148,
            0.81240985,
            0.84219775,
            0.87166406,
            0.90030504,
            0.92880418,
            0.95655449,
            0.9835349,
            1.01047155,
            1.0362095,
            1.06136425,
            1.08579312,
            1.1094819,
            1.13277995,
            1.15472826,
            1.17627308,
            1.19703503,
            1.21671694,
            1.23535898,
            1.25364313,
            1.27103891,
            1.28718029,
            1.30302865,
            1.31839527,
            1.33294817,
            1.34700935,
            1.3605727,
            1.37345513,
            1.38577122,
            1.39733504,
            1.40856397,
            1.41959619,
            1.42983624,
            1.43958467,
            1.44902176,
            1.45804831,
            1.46669568,
            1.47486938,
            1.48269965,
            1.49034339,
            1.49748214,
            1.50435106,
            1.51076426,
            1.51698915,
            1.5229097,
            1.528578,
            1.53389835,
            1.5391211,
            1.5439065,
            1.54858517,
            1.55310776,
            1.55744391,
            1.56164927,
            1.56566348,
            1.56938671,
            1.57307767,
            1.57654764,
            1.57980083,
            1.58304129,
            1.58602496,
            1.58880681,
            1.59162477,
            1.5941969,
            1.59693155,
            1.599446,
            1.60185011,
            1.60408668,
            1.60627134,
            1.60826199,
            1.61004547,
            1.61192472,
            1.61369656,
            1.61534074,
            1.61688905,
            1.61838916,
            1.61985374,
            1.62135878,
            1.62268119,
            1.62390423,
            1.62513143,
            1.62632463,
            1.6274027,
            1.62842767,
            1.62945532,
            1.6303307,
            1.63128026,
            1.63204102,
        ]
    )

    # peak normalize, get magnitude, clip lower bound
    wav = np.array(wav)
    wav = wav / abs(wav).max()
    abs_wav = abs(wav)
    abs_wav[abs_wav < eps] = eps

    # calcuate statistics
    # E[|z|]
    v1 = max(eps, abs_wav.mean())
    # E[log|z|]
    v2 = np.log(abs_wav).mean()
    # log(E[|z|]) - E[log(|z|)]
    v3 = np.log(v1) - v2

    # table interpolation
    wav_snr_idx = None
    if any(g_vals < v3):
        wav_snr_idx = np.where(g_vals < v3)[0].max()
    # handle edge cases or interpolate
    if wav_snr_idx is None:
        wav_snr = db_vals[0]
    elif wav_snr_idx == len(db_vals) - 1:
        wav_snr = db_vals[-1]
    else:
        wav_snr = db_vals[wav_snr_idx] + (v3 - g_vals[wav_snr_idx]) / (
            g_vals[wav_snr_idx + 1] - g_vals[wav_snr_idx]
        ) * (db_vals[wav_snr_idx + 1] - db_vals[wav_snr_idx])

    # Calculate SNR
    dEng = sum(wav**2)
    dFactor = 10 ** (wav_snr / 10)
    dNoiseEng = dEng / (1 + dFactor)  # Noise energy
    dSigEng = dEng * dFactor / (1 + dFactor)  # Signal energy
    snr = 10 * np.log10(dSigEng / dNoiseEng)

    return snr

class DemucsProcessor():
    def __init__(self, device: str = "cpu", model_name: str = "htdemucs"):
        self.device = device
        self.model_name = model_name
        self.model = demucs.api.Separator(model=self.model_name)

    def process(self, audio_path: str) -> Tuple[np.ndarray, int]:
        result = self.model.separate_audio_file(audio_path)[1]["vocals"]
        return result.numpy()[0], self.model.samplerate
