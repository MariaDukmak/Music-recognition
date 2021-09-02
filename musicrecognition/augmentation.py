from typing import Union, List
from pathlib import Path
from sys import platform
import torchaudio
from torch_audiomentations import Compose, LowPassFilter, HighPassFilter, AddBackgroundNoise, Gain, AddColoredNoise

# Because torch-audiomentations unfortunately has this line hardcoded:
# https://github.com/asteroid-team/torch-audiomentations/blob/a1e14c14dc714078bfda841efc8184330f4f18b7/torch_audiomentations/utils/io.py#L25
# , the following code is needed for unix users :(
if platform in {'linux', 'linux2', 'darwin'}:
    torchaudio.set_audio_backend('sox_io')


def get_augmenter(background_noise_path: Union[List[Path], List[str], Path, str]) -> Compose:
    return Compose([
        # LowPassFilter(p=.3),
        # HighPassFilter(p=.3),
        # AddBackgroundNoise(background_paths=background_noise_path, p=.9),
        # Gain(p=.5),
        # AddColoredNoise(p=.9),
        # LowPassFilter(p=.3),
        # HighPassFilter(p=.3),
    ])
