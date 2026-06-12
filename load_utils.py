import torch
import soundfile as sf
import torch.nn.functional as F
import random
import pandas as pd
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from collections import Counter
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

TARGET_SAMPLE_RATE = 32000 # 32kHz
TARGET_SECONDS = 5
TARGET_FRAMES = TARGET_SECONDS * TARGET_SAMPLE_RATE
NUM_CLASSES = 234
RANDOM_SEED = 42

__all__ = ["get_soundscapes_dataloader", "get_single_bird_dataloader", "NUM_CLASSES"]

def get_waveform(path, start=0, frames_to_read=-1):
    """
           Returns the waveform and the sample rate of the audio file.
           If the file is longer than 5 seconds, it will be randomly truncated.
           If the file is less than 5 seconds, it will be padded.
           The waveform is returned as [num_channels, frames].
       """

    audio_array, sample_rate = sf.read(path, frames=frames_to_read, start=start,dtype="float32")

    waveform = torch.from_numpy(audio_array)
    waveform = waveform.unsqueeze(0) # always mono

    num_frames = waveform.shape[1]

    if num_frames < TARGET_FRAMES:
        pad_amount = TARGET_FRAMES - num_frames
        waveform = F.pad(waveform, (0, pad_amount))
    elif num_frames > TARGET_FRAMES:
        max_start = num_frames - TARGET_FRAMES
        start_idx = random.randint(0, max_start)
        waveform = waveform[:, start_idx : start_idx + TARGET_FRAMES]

    return waveform

def wave_to_spectrogram(waveform, window_length=1024, hop_length=320, n_mel_bands=128, f_min=500, f_max=15000):
    """
        Transforms a waveform into a mel-spectrogram.
    """
    transform_spectrogram = MelSpectrogram(sample_rate=TARGET_SAMPLE_RATE, n_fft=window_length, hop_length=hop_length, n_mels=n_mel_bands, f_min=f_min, f_max=f_max)
    transform_db = AmplitudeToDB(stype='power')
    return transform_db(transform_spectrogram(waveform))

def get_spectrogram(path, start=0, frames_to_read=-1):
    return wave_to_spectrogram(get_waveform(path, start, frames_to_read))

def index_to_one_hot(target_index):
    return F.one_hot(torch.tensor(target_index), num_classes=NUM_CLASSES).to(torch.float32)

def get_single_bird_dataloader(batch_size, train_split=0.8):
    """
        Splits the data into training and validation sets and returns a DataLoader for both.
        Classes that contain only one training sample will be in the training set.
    """

    path = "../train_audio"
    dataset = DatasetFolder(root=path, loader=get_waveform, target_transform=index_to_one_hot, extensions=tuple([".ogg"]))

    targets = dataset.targets

    class_counts = Counter(targets)
    single_sample_classes = set(cls for cls, count in class_counts.items() if count == 1)

    single_sample_indices = [idx for idx, cls in enumerate(targets) if cls in single_sample_classes]
    multiple_sample_indices = [idx for idx, cls in enumerate(targets) if cls not in single_sample_classes]

    multiple_sample_targets = [targets[idx] for idx in multiple_sample_indices]

    # Stratified split --> keeps the 80% - 20% ratio of samples for all classes
    train_indices, val_indices = train_test_split(multiple_sample_indices, test_size=1 - train_split, stratify=multiple_sample_targets, random_state=RANDOM_SEED)

    train_indices += single_sample_indices

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

class SoundscapesDataset(Dataset):
    """
        Combines the label's .csv with the actual audio files to create
        a Dataset class able to provide the waveform of the audio file
        paired with its labels.
    """
    def __init__(self):
        self.df = pd.read_csv("../train_soundscapes_labels.csv").drop_duplicates(subset=["filename", "start", "primary_label"], keep="first").reset_index(drop=True)
        self.classes = pd.read_csv("../taxonomy.csv")["primary_label"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_name = row["filename"]
        start_sec = int(row["start"].split(":")[2])

        start_frame = start_sec * TARGET_SAMPLE_RATE
        frames_to_read = 5 * TARGET_SAMPLE_RATE

        path = f"../train_soundscapes/{file_name}"

        ret = get_waveform(path, frames_to_read=frames_to_read, start=start_frame)

        labels = row["primary_label"].split(";")
        labels_multi_hot_list = list(map(int, list(self.classes.isin(labels))))
        labels_multi_hot = torch.tensor(labels_multi_hot_list, dtype=torch.float32)

        return ret, labels_multi_hot

def get_soundscapes_dataloader(batch_size, train_split=0.8):
    """
        Splits the soundscapes data into training and validation sets and returns a DataLoader for both.
    """
    dataset = SoundscapesDataset()

    unique_files = dataset.df["filename"].unique()
    train_files, val_files = train_test_split(unique_files, test_size=1 - train_split, random_state=RANDOM_SEED)

    train_indices = dataset.df[dataset.df["filename"].isin(train_files)].index.tolist()
    val_indices = dataset.df[dataset.df["filename"].isin(val_files)].index.tolist()

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def visualize_spectrogram(spectrogram):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram[0].numpy(), origin='lower', aspect='auto')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train_iter, val_iter = get_soundscapes_dataloader(batch_size=2)
    for item, label in train_iter:
        print(item.shape)
        print(label.shape)
        break
