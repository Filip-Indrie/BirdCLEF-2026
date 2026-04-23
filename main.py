from train_utils import *
from load_utils import *
from spectrogram_models import *
import torch

if __name__ == '__main__':
    nets = [
        ResNet18(),
    ]

    batch_size = 64
    num_epochs = 100
    patience = 10

    lr = 1e-3
    weight_decay = 1e-3

    train_iter, val_iter = get_single_bird_dataloader(batch_size, to_spectrogram=True)

    device = try_gpu()
    print(f"Training on {torch.cuda.get_device_name(device)}")

    for net in nets:
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        train(net, train_iter, val_iter, 1, patience, optimizer, device)


