from typing import Callable
import torch
import torchaudio
import torchvision
import librosa


def get_spectrogram_func(sample_rate: int = 44100) -> torch.nn.Module:
    return torch.nn.Sequential(
        #torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, center=True, power=2.0, norm='slaney', onesided=True, n_mels=128),
        torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=256,
            melkwargs={
              'n_fft': 2048,
              'n_mels': 256,
              'hop_length': 512,
              #'mel_scale': 'htk',
            }
        ),
        torchaudio.transforms.AmplitudeToDB(top_db=80),
        torchvision.transforms.Normalize((-37.3427,), (26.6659,)),
    )


def get_librosa_func(sample_rate: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def spectrogram_func(waveforms: torch.Tensor) -> torch.Tensor:
        output = []
        for waveform in waveforms.numpy():
            mfcc = librosa.feature.mfcc(
                y=waveform, sr=sample_rate, hop_length=512, n_mfcc=13
            )
            output.append(torch.tensor(mfcc.T))
        return torch.stack(output)

    return spectrogram_func
