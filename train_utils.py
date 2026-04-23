import time
import datetime
import os
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['get_prediction', 'train', 'try_gpu']

def get_prediction(logits):
    prediction = F.sigmoid(logits)
    prediction = torch.where(prediction > 0.5, 1, 0)
    return prediction

def init_weights(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.kaiming_uniform_(layer.weight)

def evaluate_accuracy(net, data_iter, loss, device):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_hits = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)

            with torch.no_grad():
                total_loss += float(l)
                total_hits += (get_prediction(y_hat).type(y.dtype) == y).sum()
                total_samples += y.numel()

    return float(total_loss) / len(data_iter), float(total_hits) / total_samples  * 100

def train_epoch_amp(net, train_iter, loss, optimizer, scaler, device):
    # Uses automatic mixed precision

    net.train()

    # Sum of training loss, sum of training correct predictions, no. of examples
    total_loss = 0
    total_hits = 0
    total_samples = 0

    for x, y in train_iter:
        # Compute gradients and update parameters
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            y_hat = net(x)
            l = loss(y_hat, y)

        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            total_loss += float(l)
            total_hits += (get_prediction(y_hat).type(y.dtype) == y).sum()
            total_samples += y.numel()

    # Return training loss and training accuracy
    return float(total_loss) / len(train_iter), float(total_hits) / total_samples * 100

def train(net, train_iter, val_iter, num_epochs, patience, optimizer,  device, delete_old_measurements=False):
    """Train a model."""
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    best_val_accuracy = 0
    counter = 0

    # fidget with pos_weight
    # fidget with loss params so it doesn't just guess all 0s for single bird
    # it currently does this and gets 99.9% accuracy
    loss = nn.BCEWithLogitsLoss()

    net.apply(init_weights)
    net.to(device)

    net_name = type(net).__name__
    dir_name = "Measurements/" + net_name + "/"
    os.makedirs(dir_name, exist_ok=True)

    stats_file_name = dir_name + "LR_" + str(optimizer.state_dict()['param_groups'][0]['lr']) + ".txt"
    if not os.path.exists(stats_file_name) or delete_old_measurements:
        stats_file = open(stats_file_name, "w", encoding="utf-8")
        stats_file.write(str(net) + "\n\n")
    else:
        stats_file = open(stats_file_name, "a", encoding="utf-8")
        stats_file.write("\n")

    start_time = time.time()

    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        current_time = datetime.datetime.now()
        print(f"{current_time.strftime("%H:%M:%S")} Epoch {epoch + 1}")

        train_loss, train_acc = train_epoch_amp(net, train_iter, loss, optimizer, scaler, device)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)

        val_loss, val_acc = evaluate_accuracy(net, val_iter, loss, device)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)

        stats_file.write(f'Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}\n')

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    end_time = time.time()

    stats_file.write(f'Best Validation Accuracy {best_val_accuracy:.2f}, Epoch: {val_acc_all.index(best_val_accuracy) + 1}, Training Time: {end_time - start_time:.2f}s\n')

    stats_file.close()

    return train_loss_all, train_acc_all, val_loss_all, val_acc_all

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')