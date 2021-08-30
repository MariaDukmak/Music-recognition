from typing import Union, List
from pathlib import Path
import random
import torch

from musicrecognition.augmentation import get_augmenter
from musicrecognition.audio_dataloader import AudioDataloader
from musicrecognition.spectrogram import get_spectrogram_func
from musicrecognition.model import LSTMNetwork

random.seed(42)
torch.manual_seed(42)
DATA_ROOT = Path('../data')
BATCH_SIZE = 4
TEST_SIZE = 0.3  # 70% train 30% test
SONG_SAMPLE_RATE = 44100  # Most songs in the dataset seem to have a sample-rate of 44100
MIN_AUDIO_LENGTH = 10
MAX_AUDIO_LENGTH = 30
LATENT_SPACE_SIZE = 16


def get_song_paths(source_path: Union[Path, str]) -> List[Path]:
    return list(Path(source_path).glob('**/*.mp3'))


def train():
    # Train test split song paths
    songs_paths = get_song_paths(DATA_ROOT / 'songs')
    random.shuffle(songs_paths)
    test_paths, train_paths = songs_paths[:round(len(songs_paths) * TEST_SIZE)], songs_paths[round(len(songs_paths) * TEST_SIZE):]

    # Create augmentation function
    augmenter = get_augmenter(DATA_ROOT / 'background_noises')

    # Create spectrogram function
    spectrogram_func = get_spectrogram_func(SONG_SAMPLE_RATE)

    # Create the audio-data-loader
    train_loader = AudioDataloader(train_paths,
                                   augmenter,
                                   BATCH_SIZE,
                                   MIN_AUDIO_LENGTH,
                                   MAX_AUDIO_LENGTH,
                                   SONG_SAMPLE_RATE,
                                   spectrogram_func,
                                   )

    model = LSTMNetwork(128, 32, LATENT_SPACE_SIZE, 2)
    print(model)

    criterion = torch.nn.TripletMarginLoss()

    # Optimizer
    learning_rate = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for anchors, positives, negatives in train_loader:

        optimizer.zero_grad()

        latent_anchors = model(anchors)
        latent_positives = model(positives)
        latent_negatives = model(negatives)

        loss = criterion(latent_anchors, latent_positives, latent_negatives)

        loss.backward()
        optimizer.step()

        print(loss)


if __name__ == '__main__':
    train()
