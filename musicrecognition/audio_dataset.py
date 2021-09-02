from typing import List, Union
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio


class AudioDataset(Dataset):
    def __init__(self, audio_paths: List[Union[Path, str]], target_sample_rate: int):
        self.audio_paths = audio_paths
        self.target_sample_rate = target_sample_rate

    def __getitem__(self, item):
        waveform, sample_rate = torchaudio.load(self.audio_paths[item])
        if sample_rate == self.target_sample_rate:
            return waveform.mean(0)
        else:
            resample = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            return resample(waveform.mean(0, keepdim=True))[0]

    def __len__(self):
        return len(self.audio_paths)

