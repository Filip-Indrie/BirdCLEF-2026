from train_utils import *
from load_utils import *
from spectrogram_models import *

if __name__ == '__main__':
    nets = [
        ResNet18(),
    ]

    batch_size = 64
    num_epochs = 100
    patience = 15

    lr = 1e-3
    weight_decay = 1e-3
    threshold = 0.5

    train_iter, val_iter = get_single_bird_dataloader(batch_size)

    for net in nets:
        train(net, True, False, False, lr, weight_decay, threshold, train_iter, val_iter, 1, patience)


