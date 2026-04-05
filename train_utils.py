import torch
import soundfile as sf
import torch.nn.functional as F
import random
import pandas as pd
from collections import Counter
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split

TARGET_SAMPLE_RATE = 32000 # 32kHz
TARGET_SECONDS = 5
TARGET_FRAMES = TARGET_SECONDS * TARGET_SAMPLE_RATE
NUM_CLASSES = 234
RANDOM_SEED = 42

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

    return tuple([waveform, sample_rate])

# IMPROVEMENT IDEA: Use voice activity detection (VAD --> torchaudio implementations)
# to extract only the chunks where the bird actually sings
def get_single_bird_dataloader(batch_size, train_split=0.8):
    """
        Splits the data into training and validation sets and returns a DataLoader for both.
        Classes that contain only one training sample will be in the training set.
    """

    path = "../train_audio"

    dataset = DatasetFolder(root=path, loader=get_waveform, extensions=tuple([".ogg"]))

    targets = dataset.targets

    class_counts = Counter(targets)
    single_sample_classes = set(label for label, count in class_counts.items() if count == 1)

    single_sample_indices = [idx for idx, label in enumerate(targets) if label in single_sample_classes]
    multiple_sample_indices = [idx for idx, label in enumerate(targets) if label not in single_sample_classes]

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
        waveform, _ = get_waveform(path, frames_to_read=frames_to_read, start=start_frame)

        labels = row["primary_label"].split(";")
        labels_multi_hot_list = list(map(int, list(self.classes.isin(labels))))
        labels_multi_hot = torch.tensor(labels_multi_hot_list, dtype=torch.float32)

        return waveform, labels_multi_hot

def get_soundscapes_dataloader(batch_size, train_split=0.8):
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

if __name__ == "__main__":
    train_iter, val_iter = get_soundscapes_dataloader(batch_size=2)
    for wave, label in train_iter:
        print(wave, label)
        break
