from typing import Union, Callable, Tuple, List, Sequence, Optional
from pathlib import Path
import random
import torch
import torchaudio
import numpy as np


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
        self.audio_length_range = np.array([min_audio_length, max_audio_length])
        self.target_sample_rate = target_sample_rate
        self.spectrogram_func = spectrogram_func

        self.anchor_index = 0
        self.resample_kernels = {}

    def resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample (if needed) the given waveform to the standard sample-rate given at the constructor."""
        waveform = waveform.mean(0, keepdim=True)
        if sample_rate == self.target_sample_rate:
            return waveform
        else:
            if sample_rate not in self.resample_kernels:
                print(f"Encountered not before seen sample rate: {sample_rate}")
                self.resample_kernels[sample_rate] = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            return self.resample_kernels[sample_rate](waveform)

    def next_anchors(self) -> Tuple[List[torch.Tensor], List[str]]:
        """Load the next batch of anchors, also return the picked file-paths so those can be avoided in next_negatives."""
        waveforms = []
        picked_paths = []
        while len(waveforms) < self.batch_size:
            audio_path = self.audio_paths[self.anchor_index]
            try:
                waveforms.append(self.resample(*torchaudio.load(audio_path)))
                picked_paths.append(audio_path)
            except Exception as e:
                print(f"WARNING: Loading audio {audio_path} gave error: {e}")
                self.audio_paths.pop(self.anchor_index)
            finally:
                self.anchor_index = (self.anchor_index + 1) % len(self.audio_paths)
        return waveforms, picked_paths

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

    def clip_audio_samples(self, waveforms: Sequence[torch.Tensor], wanted_length: int = None) -> Tuple[List[torch.Tensor], int]:
        """
        Clip the waveforms so their lengths are the same,
        if wanted_length is not specified, a random one will be picked in the range given at the constructor
        """
        min_waveform_length = min([waveform.shape[1] for waveform in waveforms])
        if wanted_length is None:
            rand_range = np.minimum(self.audio_length_range * self.target_sample_rate, [min_waveform_length] * 2)
            wanted_length = np.random.randint(*rand_range.astype(int))
        else:
            waveforms = [torch.cat([waveform, torch.zeros(min(0, wanted_length-len(waveform)))]) for waveform in waveforms]
        begin_index = np.random.randint(min_waveform_length - wanted_length + 1)
        return [waveform[:, begin_index:begin_index+wanted_length] for waveform in waveforms], wanted_length

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Anchors
        anchor_waveforms, anchor_paths = self.next_anchors()
        anchor_waveforms, waveform_length = self.clip_audio_samples(anchor_waveforms)
        anchors_stacked = torch.stack(anchor_waveforms)

        # Positives
        positives_stacked = self.augmenter(anchors_stacked, self.target_sample_rate)

        # Negatives
        negative_waveforms, _ = self.clip_audio_samples(self.next_negatives(anchor_paths), waveform_length)
        negatives_stacked = torch.stack(negative_waveforms)

        if self.spectrogram_func:
            # Shape: [1, batch_size, height, width]
            return torch.split(
                self.spectrogram_func(
                    torch.stack(
                        [anchors_stacked[:, 0], positives_stacked[:, 0], negatives_stacked[:, 0]]
                    )
                ), 1
            )
        else:
            # Shape: [batch_size, 1, waveform_length]
            return anchors_stacked, positives_stacked, negatives_stacked
