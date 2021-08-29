from typing import Union, Callable, Tuple, List, Sequence, Optional
from pathlib import Path
import random
import torch
import torchaudio
import numpy as np


def pad_waveform(waveform: torch.Tensor, wanted_length: int) -> torch.Tensor:
    """Pad the waveform with zeros so it's at least the wanted_length."""
    return torch.cat([waveform, torch.zeros(max(0, wanted_length-waveform.shape[0]))])


def clip_length(waveforms: Sequence[torch.Tensor], wanted_length: int) -> torch.Tensor:
    """Pad and clip the waveforms to wanted_length and stack them, clipping to a smaller length uses randomness."""
    clipped = []
    for waveform in waveforms:
        padded = pad_waveform(waveform, wanted_length)
        begin_index = np.random.randint(padded.shape[0] - wanted_length)
        clipped.append(padded[begin_index:begin_index+wanted_length])
    return torch.stack(clipped)


class AudioDataloader:
    def __init__(self,
                 audio_paths: List[Union[Path, str]],
                 augmenter: Callable[[torch.Tensor, Optional[int]], torch.Tensor],
                 batch_size: int,
                 min_audio_length: float,
                 max_audio_length: float,
                 target_sample_rate: int,
                 spectrogram_func: Callable[[torch.Tensor], torch.Tensor] = None):
        self.audio_paths = audio_paths
        self.augmenter = augmenter
        self.batch_size = batch_size
        self.target_sample_rate = target_sample_rate
        self.audio_length_range = np.array([min_audio_length, max_audio_length]) * self.target_sample_rate
        self.spectrogram_func = spectrogram_func

        self.anchor_index = 0
        self.resample_kernels = {}

    def resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample (if needed) the given waveform to the standard sample-rate given at the constructor."""
        waveform = waveform.mean(0)
        if sample_rate == self.target_sample_rate:
            return waveform
        else:
            if sample_rate not in self.resample_kernels:
                print(f"Encountered not before seen sample rate: {sample_rate}")
                self.resample_kernels[sample_rate] = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            return self.resample_kernels[sample_rate](waveform.unsqueeze(0))[0]

    def next_anchors(self) -> Tuple[torch.Tensor, List[str]]:
        """Load the next batch of anchors, also return the picked file-paths so those can be avoided in next_negatives."""
        waveforms = []
        picked_paths = []
        # Load until the batch is full
        while len(waveforms) < self.batch_size:
            audio_path = self.audio_paths[self.anchor_index]
            # Try if this audio file can be loaded
            try:
                waveforms.append(self.resample(*torchaudio.load(audio_path)))
                picked_paths.append(audio_path)
            except Exception as e:
                print(f"WARNING: Loading audio {audio_path} gave error: {e}")
                self.audio_paths.pop(self.anchor_index)
            finally:
                self.anchor_index = (self.anchor_index + 1) % len(self.audio_paths)

        # Pad the waveforms the same length and stack them to get one tensor
        max_length = max([waveform.shape[0] for waveform in waveforms])
        waveforms_stacked = torch.stack([pad_waveform(waveform, max_length) for waveform in waveforms])

        return waveforms_stacked, picked_paths

    def next_negatives(self, avoid_paths: Sequence[Union[Path, str]]) -> List[torch.Tensor]:
        """Load the next batch of negatives."""
        waveforms = []
        for avoid_path in avoid_paths:
            while True:
                audio_path = random.choice(self.audio_paths)
                if audio_path != avoid_path:
                    try:
                        waveforms.append(self.resample(*torchaudio.load(audio_path)))
                    except Exception as e:
                        print(f"WARNING: Loading audio {audio_path} gave error: {e}")
                        continue
                    break
        return waveforms

    def random_length(self, max_length: int) -> int:
        """Get random length between range given at the constructor but maximum the given max_length parameter."""
        rand_range = np.minimum(self.audio_length_range, [max_length] * 2)
        return np.random.randint(*rand_range)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Anchors
        anchors, anchor_paths = self.next_anchors()

        # Choose random length for the positives and negatives
        random_length = self.random_length(anchors.shape[1])

        # Positives
        positives = self.augmenter(clip_length(anchors, random_length).unsqueeze(1), self.target_sample_rate)[:, 0]

        # Negatives
        negatives = clip_length(self.next_negatives(anchor_paths), random_length)

        if self.spectrogram_func:
            # Shape: [1, batch_size, height, width]
            return torch.split(self.spectrogram_func(torch.stack([anchors, positives, negatives])), 1)
        else:
            # Shape: [batch_size, waveform_length]
            return anchors, positives, negatives
