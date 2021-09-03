from typing import Union, List
from pathlib import Path
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from musicrecognition.augmentation import get_augmenter
from musicrecognition.spectrogram import get_spectrogram_func
from musicrecognition.model import LSTMNetwork, SimpleNet
from musicrecognition.audio_dataset import AudioDataset
from musicrecognition.data_collate import create_collate_fn

random.seed(42)
torch.manual_seed(42)
DATA_ROOT = Path('../data')
BATCH_SIZE = 20
TEST_SIZE = 0.3  # 70% train 30% test
SONG_SAMPLE_RATE = 44100  # Most songs in the dataset seem to have a sample-rate of 44100
MIN_AUDIO_LENGTH = 10
MAX_AUDIO_LENGTH = 30
LATENT_SPACE_SIZE = 32


def get_song_paths(source_path: Union[Path, str]) -> List[Path]:
    return list(Path(source_path).glob('**/*.mp3'))


def train():
    # Train test split song paths
    songs_paths = get_song_paths(DATA_ROOT / 'songs')
    random.shuffle(songs_paths)
    test_paths, train_paths = songs_paths[:round(len(songs_paths) * TEST_SIZE)], songs_paths[round(len(songs_paths) * TEST_SIZE):]

    train_set = AudioDataset(train_paths, SONG_SAMPLE_RATE)

    # Create augmentation function
    augmenter = get_augmenter(DATA_ROOT / 'background_noises')

    # Create spectrogram function
    spectrogram_func = get_spectrogram_func(SONG_SAMPLE_RATE)

    collate_fn = create_collate_fn(augmenter, MIN_AUDIO_LENGTH, MAX_AUDIO_LENGTH, SONG_SAMPLE_RATE, spectrogram_func)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=12, collate_fn=collate_fn)

    device = torch.device('cuda:1')
    #writer = SummaryWriter(comment=input("Test name: "))

    model = LSTMNetwork(128, 64, LATENT_SPACE_SIZE, 2).to(device)
    # model = SimpleNet(128, LATENT_SPACE_SIZE, 10).to(device)

    criterion = torch.nn.TripletMarginLoss()

    # Optimizer
    learning_rate = 0.000000000000001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    anchors, positives, negatives = next(iter(train_loader))
    anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
    epoch = 0
    while True:
        epoch += 1
        latent_anchors = model(anchors.transpose(1, 2))
        latent_positives = model(positives.transpose(1, 2))
        latent_negatives = model(negatives.transpose(1, 2))
        loss = criterion(latent_anchors, latent_positives, latent_negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_index = epoch * BATCH_SIZE
        print(f"{time_index} loss {loss.item()}")

    # Training loop
    for epoch in range(100):
        for n_iter, (anchors, positives, negatives) in enumerate(train_loader):
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            latent_anchors = model(anchors.transpose(1, 2))
            latent_positives = model(positives.transpose(1, 2))
            latent_negatives = model(negatives.transpose(1, 2))

            loss = criterion(latent_anchors, latent_positives, latent_negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time_index = epoch * len(train_set) + n_iter * BATCH_SIZE
            print(f"{time_index} loss {loss.item()}")
            writer.add_scalar('Loss/train', loss.item(), time_index)
        torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    train()
