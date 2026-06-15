from train_utils import *
from load_utils import *
from spectrogram_models import *
import os

if __name__ == '__main__':
    nets = [
        ResNet18(),
    ]

    batch_size = 256
    positive_label_smoothing = 0.1

    num_epochs_train = 100
    patience_train = 15
    lr_train = 3e-3
    weight_decay_train = 0.01
    threshold_train = 0.85

    num_epochs_fine_tune = 50
    patience_fine_tune = 10
    lr_fine_tune = 1e-4
    weight_decay_fine_tune = 0.01
    threshold_fine_tune = 0.7

    device = try_gpu()

    train_iter_train, val_iter_train, pos_weights_train = get_single_bird_dataloader(device, batch_size, pos_weights_clamp_max=100)
    train_iter_fine_tune, val_iter_fine_tune, pos_weights_fine_tune = get_soundscapes_dataloader(device, batch_size, pos_weights_clamp_max=25)

    try:
        for net in nets:
            train_model(
                device, net, True, False,
                lr_train, weight_decay_train, positive_label_smoothing, threshold_train, True,
                train_iter_train, val_iter_train, pos_weights_train,
                num_epochs_train, patience_train, save_weights=True,
                save_folder="First Full Training Loop (Single Birds)"
            )

            train_model(
                device, net, True, True,
                lr_fine_tune, weight_decay_fine_tune, positive_label_smoothing, threshold_fine_tune, False,
                train_iter_fine_tune, val_iter_fine_tune, pos_weights_fine_tune,
                num_epochs_fine_tune, patience_fine_tune, save_weights=True,
                save_folder="First Full Training Loop (Soundscapes)"
            )

    except Exception as e:
        with open("error.txt", "w") as f:
            print('ERROR!')
            f.write(str(e))
    finally:
        # os.system("shutdown /s /t 5")
        pass