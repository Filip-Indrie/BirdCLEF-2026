import torch
import soundfile as sf
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from collections import Counter
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Subset, Dataset, RandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from train_utils import try_gpu

load_dotenv()
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE"))
TARGET_SECONDS = int(os.getenv("TARGET_SECONDS"))
TARGET_FRAMES = TARGET_SECONDS * TARGET_SAMPLE_RATE
NUM_CLASSES = int(os.getenv("NUM_CLASSES"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED"))
DATASET_FOLDER = os.getenv("DATASET_FOLDER")

__all__ = ["get_soundscapes_dataloader", "get_single_bird_dataloader"]

def get_waveform(path: str, start: int | None = None):
    """
           Returns the waveform and the sample rate of the audio file.
           If the file is longer than 5 seconds, it will be randomly truncated.
           If the file is less than 5 seconds, it will be padded.
           The waveform is returned as [num_channels, frames].
       """

    if start is None:
        audio_seconds = round(sf.info(path).duration)
        audio_frames = audio_seconds * TARGET_SAMPLE_RATE
        if audio_frames <= TARGET_FRAMES:
            audio_array, sample_rate = sf.read(path, frames=-1, start=0, dtype="float32")

        else:
            max_start = audio_frames - TARGET_FRAMES
            start_idx = random.randint(0, max_start)

            audio_array, sample_rate = sf.read(path, frames=TARGET_FRAMES, start=start_idx, dtype="float32")

        waveform = torch.from_numpy(audio_array)
        waveform = waveform.unsqueeze(0)  # always mono

        pad_amount = TARGET_FRAMES - waveform.shape[1]
        if pad_amount: waveform = F.pad(waveform, (0, pad_amount))

    else:
        audio_array, sample_rate = sf.read(path, frames=TARGET_FRAMES, start=start, dtype="float32")

        waveform = torch.from_numpy(audio_array)
        waveform = waveform.unsqueeze(0)  # always mono

    return waveform

def index_to_one_hot(target_index):
    return F.one_hot(torch.tensor(target_index), num_classes=NUM_CLASSES).to(torch.float32)

def calculate_pos_weights_single_bird(device, train_indices, targets, clamp_max = 100):
    pos_counts = torch.zeros(NUM_CLASSES, device=device)
    total_samples = len(train_indices)

    for idx in train_indices:
        label = targets[idx]
        pos_counts[label] += 1

    pos_counts = torch.clamp(pos_counts, min=1.0)
    neg_counts = total_samples - pos_counts
    pos_weights = (neg_counts / pos_counts).to(device)
    pos_weights = pos_weights.clamp(max=clamp_max)

    return pos_weights

def get_single_bird_dataloader(device, batch_size, train_split=0.8, pos_weights_clamp_max = 100):
    """
        Splits the data into training and validation sets and returns a DataLoader for both.
        Classes that contain only one training sample will be in the training set.
    """

    path = f"{DATASET_FOLDER}/train_audio"
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

    train_pos_weights = calculate_pos_weights_single_bird(device, train_indices, targets, clamp_max = pos_weights_clamp_max)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=7, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=7, pin_memory=True, prefetch_factor=2
    )

    return train_loader, val_loader, train_pos_weights

class SoundscapesDataset(Dataset):
    """
        Combines the label's .csv with the actual audio files to create
        a Dataset class able to provide the waveform of the audio file
        paired with its labels.
    """
    def __init__(self):
        df_scientists = pd.read_csv(f"{DATASET_FOLDER}/train_soundscapes_labels.csv").drop_duplicates(subset=["filename", "start", "primary_label"], keep="first")
        df_birdnet = pd.read_csv(f"{DATASET_FOLDER}/birdnet_predicted_labels.csv")
        df_perch = pd.read_csv(f"{DATASET_FOLDER}/perch_predicted_labels.csv")

        human_labeled_files = df_scientists['filename'].unique()

        df_birdnet = df_birdnet[~df_birdnet['filename'].isin(human_labeled_files)]
        df_perch = df_perch[~df_perch['filename'].isin(human_labeled_files)]
        df_machines = pd.concat([df_birdnet, df_perch])

        df_machines_merged = df_machines.groupby(['filename', 'start', 'end'])['primary_label'].apply(
            lambda x: 'nocall' if all(l == 'nocall' for l in x)
            else ';'.join(sorted(set(';'.join(x[x != 'nocall']).split(';'))))
        ).reset_index()

        self.df = pd.concat([df_scientists, df_machines_merged], ignore_index=True).reset_index(drop=True)
        self.classes = pd.read_csv(f"{DATASET_FOLDER}/taxonomy.csv")["primary_label"]

        def create_multi_hot(label_string):
            labels = label_string.split(";")
            labels_multi_hot_list = list(map(int, list(self.classes.isin(labels))))
            return torch.tensor(labels_multi_hot_list, dtype=torch.float32)

        self.precomputed_labels = self.df["primary_label"].apply(create_multi_hot).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_name = row["filename"]
        start_sec = int(row["start"].split(":")[2])

        start_frame = start_sec * TARGET_SAMPLE_RATE

        path = f"{DATASET_FOLDER}/train_soundscapes/{file_name}"

        ret = get_waveform(path, start=start_frame)

        labels_multi_hot = self.precomputed_labels[idx]

        return ret, labels_multi_hot

def calculate_pos_weights_soundscapes(device, train_df, clamp_max = 100):
    classes = pd.read_csv(f"{DATASET_FOLDER}/taxonomy.csv")["primary_label"]

    total_samples = len(train_df)

    class_counts = Counter()
    for labels_str in train_df['primary_label']:
        labels = labels_str.split(";")
        for label in labels:
            if label in classes.values:
                class_counts[label] += 1

    pos_weights = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)

    for idx, class_name in enumerate(classes):
        pos_count = class_counts.get(class_name, 0)

        if pos_count == 0:
            pos_weights[idx] = 1.0
        else:
            neg_count = total_samples - pos_count
            weight = neg_count / pos_count
            pos_weights[idx] = weight

    pos_weights = pos_weights.clamp(max=clamp_max)
    return pos_weights

def get_soundscapes_dataloader(
        device, batch_size, train_split=0.8,
        train_samples_per_epoch: int | None = None, validation_total_samples: int | None = None,
        pos_weights_clamp_max = 25,
):
    """
        Splits the soundscapes data into training and validation sets and returns a DataLoader for both.
    """
    dataset = SoundscapesDataset()

    unique_files = dataset.df["filename"].unique()

    train_files, val_files = train_test_split(unique_files, test_size=1 - train_split, random_state=RANDOM_SEED)

    train_indices = dataset.df[dataset.df["filename"].isin(train_files)].index.tolist()
    val_indices = dataset.df[dataset.df["filename"].isin(val_files)].index.tolist()

    train_df = dataset.df.iloc[train_indices]

    bird_indices = train_df[train_df['primary_label'] != 'nocall'].index.tolist()
    nocall_indices = train_df[train_df['primary_label'] == 'nocall'].index.tolist()

    target_nocall_count = len(bird_indices) * 2

    if len(nocall_indices) > target_nocall_count:
        np.random.seed(RANDOM_SEED)
        nocall_indices = np.random.choice(nocall_indices, size=target_nocall_count, replace=False).tolist()

    balanced_train_indices = bird_indices + nocall_indices

    balanced_train_df = dataset.df.iloc[balanced_train_indices]
    train_pos_weights = calculate_pos_weights_soundscapes(device, balanced_train_df, clamp_max=pos_weights_clamp_max)

    target_val_count = int(len(balanced_train_indices) * ((1 - train_split) / train_split))
    val_size = validation_total_samples if validation_total_samples is not None else target_val_count

    if val_size < len(val_indices):
        np.random.seed(RANDOM_SEED)
        val_indices = np.random.choice(val_indices, size=val_size, replace=False).tolist()

    train_dataset = Subset(dataset, balanced_train_indices)
    val_dataset = Subset(dataset, val_indices)

    if train_samples_per_epoch is not None and train_samples_per_epoch < len(train_dataset):
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=train_samples_per_epoch
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=7, pin_memory=True, prefetch_factor=2
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=7, pin_memory=True, prefetch_factor=2
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=7, pin_memory=True, prefetch_factor=2
    )

    return train_loader, val_loader, train_pos_weights

def visualize_spectrogram(spectrogram):
    plt.figure(figsize=5)
    plt.imshow(spectrogram[0].numpy(), origin='lower', aspect='auto')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    _device = try_gpu()

    # 28480 train samples | 7168 test samples
    train_iter, val_iter, _train_pos_weights = get_single_bird_dataloader(_device, batch_size=64, train_split=0.8)

    # train_iter, val_iter, _train_pos_weights = get_soundscapes_dataloader(
    #     _device, batch_size=64, train_split=0.8, train_samples_per_epoch=28500, validation_total_samples=7200
    # )
    print(len(train_iter))
    print(len(val_iter))
    for item, _label in train_iter:
        print(item.shape)
        print(_label.shape)
        break
