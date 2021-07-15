"""
Taken from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#simulating-a-phone-recoding
"""

from typing import Tuple
import torch
import torchaudio
import math
import os
import requests


_SAMPLE_DIR = "_sample_data"
os.makedirs(_SAMPLE_DIR, exist_ok=True)
SAMPLE_RIR_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/room-response/rm1/impulse/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo.wav"
SAMPLE_RIR_PATH = os.path.join(_SAMPLE_DIR, "rir.wav")
SAMPLE_NOISE_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/distractors/rm1/babb/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"
SAMPLE_NOISE_PATH = os.path.join(_SAMPLE_DIR, "bg.wav")

uri = [
    (SAMPLE_RIR_URL, SAMPLE_RIR_PATH),
    (SAMPLE_NOISE_URL, SAMPLE_NOISE_PATH),
]
for url, path in uri:
    if not os.path.exists(path):
        with open(path, 'wb') as file_:
            file_.write(requests.get(url).content)


def _get_sample(path, resample=None):
    effects = [
        ["remix", "1"]
    ]
    if resample:
        effects.extend([
          ["lowpass", f"{resample // 2}"],
          ["rate", f'{resample}'],
        ])
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def get_rir_sample(*, resample=None, processed=False):
    rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
    if not processed:
        return rir_raw, sample_rate
    rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    return rir, sample_rate


def get_noise_sample(*, resample=None):
    return _get_sample(SAMPLE_NOISE_PATH, resample=resample)


def simulate_phone_recording(waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
    # Apply RIR
    rir, _ = get_rir_sample(resample=sample_rate, processed=True)
    waveform_ = torch.nn.functional.pad(waveform, (rir.shape[1]-1, 0))
    waveform = torch.nn.functional.conv1d(waveform_[None, ...], rir[None, ...])[0]

    # Add background noise
    # Because the noise is recorded in the actual environment, we consider that
    # the noise contains the acoustic feature of the environment. Therefore, we add
    # the noise after RIR application.
    noise, _ = get_noise_sample(resample=sample_rate)
    noise = noise[:, :waveform.shape[1]]

    snr_db = 8
    scale = math.exp(snr_db / 10) * noise.norm(p=2) / waveform.norm(p=2)
    waveform = (scale * waveform + noise) / 2

    # Apply filtering and change sample rate
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform,
        sample_rate,
        effects=[
            ["lowpass", "4000"],
            ["compand", "0.02,0.05", "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8", "-7", "0.05"],
            ["rate", "8000"],
        ],
    )

    # Apply telephony codec
    waveform = torchaudio.functional.apply_codec(waveform, sample_rate, format="gsm")

    return waveform, sample_rate
