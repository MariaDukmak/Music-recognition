from typing import List, Tuple, Callable, Optional, Sequence
import torch
import numpy as np


def pad_waveform(waveform: torch.Tensor, wanted_length: int) -> torch.Tensor:
    """Pad the waveform with zeros so it's at least the wanted_length."""
    return torch.cat([waveform, torch.zeros(max(0, wanted_length-waveform.shape[0]))])


def clip_length(waveforms: Sequence[torch.Tensor], wanted_length: int) -> torch.Tensor:
    """Pad and clip the waveforms to wanted_length and stack them, clipping to a smaller length uses randomness."""
    clipped = []
    for waveform in waveforms:
        begin_index = np.random.randint(waveform.shape[0] - wanted_length + 1)
        clipped.append(waveform[begin_index:begin_index+wanted_length])
    return torch.stack(clipped)


def create_collate_fn(augmenter: Callable[[torch.Tensor, Optional[int]], torch.Tensor],
                      min_audio_length: float,
                      max_audio_length: float,
                      sample_rate: int,
                      spectrogram_func: Callable[[torch.Tensor], torch.Tensor] = None):

    audio_length_range = (np.array([min_audio_length, max_audio_length]) * sample_rate).astype(int)

    def collate_fn(input_batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Anchors
            max_length = max([waveform.shape[0] for waveform in input_batch])
            anchors = torch.stack([pad_waveform(waveform, max_length) for waveform in input_batch])

            # Choose random length for the positives and negatives
            min_length = min([waveform.shape[0] for waveform in input_batch])
            random_length = np.random.randint(*np.minimum(audio_length_range, [min_length] * 2))

            # Positives
            positives = augmenter(clip_length(input_batch, random_length).unsqueeze(1), sample_rate)[:, 0]

            if spectrogram_func:
                # Shape: [batch_size, height, width]
                anchor_spec = spectrogram_func(anchors)
                positive_spec = spectrogram_func(positives)
                return anchor_spec, positive_spec, positive_spec.roll(1, 0)

            else:
                # Shape: [batch_size, waveform_length]
                return anchors, positives, positives.roll(1, 0)

    return collate_fn
