from torch import nn
import torchaudio
import torchvision


def get_spectrogram_func(sample_rate: int = 44100) -> nn.Module:
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate),
        torchaudio.transforms.AmplitudeToDB(),
        torchvision.transforms.Normalize((-27.2581,), (37.0171,)),
    )
