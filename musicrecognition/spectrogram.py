from torch import nn
import torchaudio


def get_spectrogram_func(sample_rate: int = 44100) -> nn.Module:
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate),
        torchaudio.transforms.AmplitudeToDB()
    )
