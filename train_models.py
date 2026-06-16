from train_utils import *
from load_utils import *
from spectrogram_models import *
import os

if __name__ == '__main__':
    nets = [
        ResNet18(),
    ]

    batch_size = 256
    num_epochs = 100
    patience = 15

    lr = 3e-3
    weight_decay = 0.01
    threshold = 0.85
    positive_label_smoothing = 0.1

    device = try_gpu()

    train_iter, val_iter, pos_weights = get_single_bird_dataloader(device, batch_size, pos_weights_clamp_max=100)

    try:
        for net in nets:
            train_model(
                device, net, True, False,
                lr, weight_decay, positive_label_smoothing, threshold, True,
                train_iter, val_iter, pos_weights, num_epochs, patience, save_weights=True,
                save_folder="Single Bird Training"
            )
    except Exception as e:
        with open("error.txt", "w") as f:
            print('ERROR!')
            f.write(str(e))
    finally:
        # os.system("shutdown /s /t 5")
        pass