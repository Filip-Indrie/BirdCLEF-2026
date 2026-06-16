import torch
import torchmetrics
import tqdm
import json
import os
from dotenv import load_dotenv
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from matplotlib import pyplot as plt

from load_utils import get_soundscapes_dataloader, get_single_bird_dataloader
from spectrogram_models import ResNet18
from train_utils import try_gpu, wave_to_spectrogram

load_dotenv()
NUM_CLASSES = int(os.getenv("NUM_CLASSES"))
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE"))

def init_spectrogram(device, window_length=1024, hop_length=320, n_mel_bands=128, f_min=500, f_max=15000):
    transform_spectrogram = MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE, n_fft=window_length, hop_length=hop_length,
        n_mels=n_mel_bands, f_min=f_min, f_max=f_max
    ).to(device)
    transform_db = AmplitudeToDB(stype='power').to(device)
    return lambda wave: transform_db(transform_spectrogram(wave))

def evaluate_model(net, data_iter, spectrogram_transform, device):
    net.eval()

    validation_loop = tqdm.tqdm(data_iter, desc="Validation Batches")

    probs_list = []
    labels_list = []

    with torch.no_grad():
        for wave, labels in validation_loop:
            wave, labels = wave.to(device), labels.to(device)

            model_input = wave if spectrogram_transform is None else wave_to_spectrogram(wave, spectrogram_transform)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = net(model_input)

            probs = torch.sigmoid(logits)
            labels_int = labels.long()

            probs_list.append(probs.cpu())
            labels_list.append(labels_int.cpu())

    return torch.cat(probs_list), torch.cat(labels_list)

def plot_scores(scores):
    x_axis = scores['thresholds']
    for label_name, y_values in scores.items():
        if label_name == "thresholds": continue
        plt.plot(x_axis, y_values, marker='o', label=label_name)

    plt.title("Model Scores")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    _device = try_gpu()

    model = ResNet18()
    weights_path = "./Measurements/ResNet18/Soundscapes ALL Fine-Tune (OneCycleLR)/weights.pth"
    state_dict = torch.load(weights_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(_device)

    spectrogram_model = True
    _spectrogram_transform = init_spectrogram(_device) if spectrogram_model else None

    batch_size = 256

    _, val_loader, _ = get_soundscapes_dataloader(_device, batch_size)

    _probs, _labels = evaluate_model(model, val_loader, _spectrogram_transform, _device)

    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    results = {
        "thresholds": thresholds,
        "f1": [],
        "precision": [],
        "recall": [],
    }

    for _threshold in thresholds:
        f1_metric = torchmetrics.classification.MultilabelF1Score(
            num_labels=NUM_CLASSES, average='macro', threshold=_threshold
        )
        precision_metric = torchmetrics.classification.MultilabelPrecision(
            num_labels=NUM_CLASSES, average='macro', threshold=_threshold
        )
        recall_metric = torchmetrics.classification.MultilabelRecall(
            num_labels=NUM_CLASSES, average='macro', threshold=_threshold
        )

        f1_metric.update(_probs, _labels)
        precision_metric.update(_probs, _labels)
        recall_metric.update(_probs, _labels)

        results["f1"].append(f1_metric.compute().item())
        results["precision"].append(precision_metric.compute().item())
        results["recall"].append(recall_metric.compute().item())

    plot_scores(results)

    with open("threshold_measurements.json", "w") as f:
        f.write(json.dumps(results, indent=4))