from typing import Union, List
from pathlib import Path
import random

from musicrecognition.augmentation import get_augmenter
from musicrecognition.audio_dataloader import AudioDataloader

random.seed(42)
DATA_ROOT = Path('../data')
BATCH_SIZE = 4
TEST_SIZE = 0.3  # 70% train 30% test
SONG_SAMPLE_RATE = 44100  # Most songs in the dataset seem to have a sample-rate of 44100
MIN_AUDIO_LENGTH = 10
MAX_AUDIO_LENGTH = 30


def get_song_paths(source_path: Union[Path, str]) -> List[Path]:
    return list(Path(source_path).glob('**/*.mp3'))


def train():
    # Train test split song paths
    songs_paths = get_song_paths(DATA_ROOT / 'songs')
    random.shuffle(songs_paths)
    test_paths, train_paths = songs_paths[:round(len(songs_paths) * TEST_SIZE)], songs_paths[round(len(songs_paths) * TEST_SIZE):]

    # Create augmentation function
    augmenter = get_augmenter(DATA_ROOT / 'background_noises')

    # Create the audio-data-loader
    train_loader = AudioDataloader(train_paths, augmenter, BATCH_SIZE, MIN_AUDIO_LENGTH, MAX_AUDIO_LENGTH, SONG_SAMPLE_RATE)

    # Training loop
    for anchors, positives, negatives in train_loader:
        print(anchors.shape, positives.shape, negatives.shape)
        # ... training code needed ...


if __name__ == '__main__':
    train()
