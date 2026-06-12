import time
import datetime
import os
import torch
import torchmetrics
from torch import nn
import tqdm
import json
import copy
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking

from load_utils import NUM_CLASSES, TARGET_SAMPLE_RATE

__all__ = ['train']

def init_weights(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.kaiming_uniform_(layer.weight)

def init_spectrogram(device, window_length=1024, hop_length=320, n_mel_bands=128, f_min=500, f_max=15000):
    transform_spectrogram = MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE, n_fft=window_length, hop_length=hop_length,
        n_mels=n_mel_bands, f_min=f_min, f_max=f_max
    ).to(device)
    transform_db = AmplitudeToDB(stype='power').to(device)
    return lambda wave: transform_db(transform_spectrogram(wave))

def wave_to_spectrogram(waveform, spectrogram_transform):
    """
        Transforms a waveform into a mel-spectrogram.
    """
    return spectrogram_transform(waveform)

def evaluate_accuracy(net, data_iter, loss, spectrogram_transform, f1_metric, precision_metric, recall_metric, device):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode

    total_loss = 0.0

    validation_loop = tqdm.tqdm(data_iter, desc="Validation batches")

    with torch.no_grad():
        for wave, labels in validation_loop:
            wave, labels = wave.to(device), labels.to(device)

            model_input = wave if spectrogram_transform is None else wave_to_spectrogram(wave, spectrogram_transform)

            logits = net(model_input)
            l = loss(logits, labels)
            total_loss += float(l)

            probs = torch.sigmoid(logits)
            labels_int = labels.long()

            f1_metric.update(probs, labels_int)
            precision_metric.update(probs, labels_int)
            recall_metric.update(probs, labels_int)

    avg_loss = total_loss / len(data_iter)
    macro_f1 = f1_metric.compute().item()
    macro_precision = precision_metric.compute().item()
    macro_recall = recall_metric.compute().item()

    f1_metric.reset()
    precision_metric.reset()
    recall_metric.reset()

    return avg_loss, macro_f1, macro_precision, macro_recall

def train_epoch_amp(net, train_iter, loss, spectrogram_transform, augment_pipeline, optimizer, scaler, f1_metric, precision_metric, recall_metric, device):
    # Uses automatic mixed precision

    net.train()

    total_loss = 0.0

    training_loop = tqdm.tqdm(train_iter, desc="Training batches")

    for wave, labels in training_loop:
        wave, labels = wave.to(device), labels.to(device)

        with torch.no_grad():
            if spectrogram_transform is not None:
                model_input = wave_to_spectrogram(wave, spectrogram_transform)
                model_input = augment_pipeline(model_input)
            else:
                model_input = wave

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = net(model_input)
            l = loss(logits, labels)

        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            total_loss += float(l)

            probs = torch.sigmoid(logits)
            labels_int = labels.long()

            f1_metric.update(probs, labels_int)
            precision_metric.update(probs, labels_int)
            recall_metric.update(probs, labels_int)

    avg_loss = total_loss / len(train_iter)
    macro_f1 = f1_metric.compute().item()
    macro_precision = precision_metric.compute().item()
    macro_recall = recall_metric.compute().item()

    f1_metric.reset()
    precision_metric.reset()
    recall_metric.reset()

    return avg_loss, macro_f1, macro_precision, macro_recall

def train(net, spectrogram_model: bool, fine_tune: bool, pre_trained: bool, lr, weight_decay, threshold: float, train_iter, val_iter, num_epochs, patience, delete_old_measurements: bool = False, save_json: bool = False, save_weights: bool = False):
    """Train a model."""

    train_loss_all = []
    train_f1_all = []
    train_precision_all = []
    train_recall_all = []
    val_loss_all = []
    val_f1_all = []
    val_precision_all = []
    val_recall_all = []

    device = try_gpu()
    print(f"Training on {torch.cuda.get_device_name(device)}")

    if spectrogram_model:
        spectrogram_transform = init_spectrogram(device)
        augment_pipeline = nn.Sequential(
            TimeMasking(time_mask_param=30),
            FrequencyMasking(freq_mask_param=15)
        ).to(device)
    else:
        spectrogram_transform = None
        augment_pipeline = None

    f1_metric = torchmetrics.classification.MultilabelF1Score(num_labels=NUM_CLASSES, average='macro', threshold=threshold).to(device)
    precision_metric = torchmetrics.classification.MultilabelPrecision(num_labels=NUM_CLASSES, average='macro', threshold=threshold).to(device)
    recall_metric = torchmetrics.classification.MultilabelRecall(num_labels=NUM_CLASSES, average='macro', threshold=threshold).to(device)

    best_val_f1 = 0
    best_val_f1_epoch = 0
    best_weights = None
    counter = 0

    pos_counts = torch.zeros(NUM_CLASSES, device=device)
    total_samples = 0

    for _, labels in train_iter:
        labels = labels.to(device)
        pos_counts += labels.sum(dim=0)
        total_samples += labels.size(0)

    pos_counts = torch.clamp(pos_counts, min=1.0)
    neg_counts = total_samples - pos_counts
    pos_weights = (neg_counts / pos_counts).to(device)

    print('Finished calculating pos_weights')

    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    if not pre_trained:
        net.apply(init_weights)
    net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 3)
    scaler = torch.amp.GradScaler()

    net_name = type(net).__name__
    dir_name = "Measurements/" + net_name + "/"
    os.makedirs(dir_name, exist_ok=True)

    mode_str = "Fine_tune" if fine_tune else "Train"
    lr_str = "_LR_" + str(optimizer.state_dict()['param_groups'][0]['lr'])
    wd_str = "_WD_" + str(optimizer.state_dict()['param_groups'][0]['weight_decay'])
    threshold_str = f"_Threshold_{threshold: .2f}"
    stats_base_file_name = dir_name + mode_str + lr_str + wd_str + threshold_str

    text_file = stats_base_file_name + ".txt"
    if not os.path.exists(text_file) or delete_old_measurements:
        stats_file = open(text_file, "w", encoding="utf-8")
        stats_file.write(str(net) + "\n\n")
    else:
        stats_file = open(text_file, "a", encoding="utf-8")
        stats_file.write("\n")

    start_time = time.time()

    for epoch in range(num_epochs):
        current_time = datetime.datetime.now()
        epoch_start_time = time.time()
        epoch_string = f"{current_time.strftime("%H:%M:%S")} Epoch {epoch + 1}"
        print(f"{current_time.strftime("%H:%M:%S")} Epoch {epoch + 1}")
        stats_file.write(epoch_string + "\n")

        train_loss, train_f1, train_precision, train_recall = train_epoch_amp(
            net, train_iter, loss, spectrogram_transform, augment_pipeline, optimizer, scaler, f1_metric, precision_metric, recall_metric, device
        )
        train_loss_all.append(train_loss)
        train_f1_all.append(train_f1)
        train_precision_all.append(train_precision)
        train_recall_all.append(train_recall)

        val_loss, val_f1, val_precision, val_recall = evaluate_accuracy(
            net, val_iter, loss, spectrogram_transform, f1_metric, precision_metric, recall_metric, device
        )

        val_loss_all.append(val_loss)
        val_f1_all.append(val_f1)
        val_precision_all.append(val_precision)
        val_recall_all.append(val_recall)

        # ONLY STEP SCHEDULER HERE IF USING `ReduceLROnPlateau`
        lr_scheduler.step(val_loss)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        stats_string = \
        f"""\
        Train stats:
        Loss: {train_loss: .4f} | F1: {train_f1: .4f} | Precision: {train_precision: .4f} | Recall: {train_recall: .4f}
        Validation stats:
        Loss: {val_loss: .4f} | F1: {val_f1: .4f} | Precision: {val_precision: .4f} | Recall: {val_recall: .4f}
        Epoch time: {int(epoch_time)} seconds = {int(epoch_time) / 60: .2f} minutes
        ===============================================================================================================\
        """

        print(stats_string)
        stats_file.write(stats_string + "\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_f1_epoch = epoch
            best_weights = copy.deepcopy(net.state_dict())
            counter = 0

        counter += 1

        if counter >= patience:
            break

    end_time = time.time()

    train_stats_string = f'Best Validation F1 {best_val_f1:.4f}, Epoch: {best_val_f1_epoch + 1}, Training Time: {end_time - start_time:.2f}s'
    print(train_stats_string)
    stats_file.write(train_stats_string + "\n")
    stats_file.close()

    if save_json:
        with open(stats_base_file_name + ".json", "w", encoding="utf-8") as json_file:
            history = {
                "num_params": sum(p.numel() for p in net.parameters()),
                "training_time": end_time - start_time,
                "train_loss": train_loss_all,
                "train_f1": train_f1_all,
                "train_precision": train_precision_all,
                "train_recall": train_recall_all,
                "val_loss": val_loss_all,
                "val_f1": val_f1_all,
                "val_precision": val_precision_all,
                "val_recall": val_recall_all,
            }
            json.dump(history, json_file, indent=4)

    if save_weights:
        torch.save(best_weights, stats_base_file_name + ".pth")

    return train_loss_all, train_f1_all, val_loss_all, val_f1_all

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')