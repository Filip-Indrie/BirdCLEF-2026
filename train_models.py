from train_utils import *
from load_utils import *
from spectrogram_models import *
import os

if __name__ == '__main__':
    nets = [
        # net, pre_trained, spectrogram_transform
        # (ResNet18(), False, None),
        (EfficientNetB1(), True, get_spectrogram_transform(240, 240)),
    ]

    batch_size = 256
    num_epochs = 100
    patience = 15

    lr = 5e-3
    weight_decay = 0.01
    threshold = 0.7
    positive_label_smoothing = 0.1

    device = try_gpu()

    train_iter, val_iter, pos_weights = get_single_bird_dataloader(device, batch_size, pos_weights_clamp_max=100)

    try:
        for net, pre_trained, spectrogram_transform in nets:
            epochs = num_epochs if not pre_trained else num_epochs // 2
            train_model(
                device, net, True, pre_trained, spectrogram_transform,
                lr, weight_decay, positive_label_smoothing, threshold, True,
                train_iter, val_iter, pos_weights, num_epochs, patience, save_weights=True, save_json=True,
                save_folder="Training (240x240, lr=1e-5, wd=0.05)/Single Birds"
            )
    except Exception as e:
        with open("error.txt", "w") as f:
            print('ERROR!')
            f.write(str(e))
    finally:
        # os.system("shutdown /s /t 5")
        pass