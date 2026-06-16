from train_utils import *
from load_utils import *
from spectrogram_models import *

if __name__ == '__main__':
    nets = [
        # model, weights_file
        (ResNet18(), "./Measurements/ResNet18/Single Bird Training (OneCycleLR)/weights.pth"),
    ]

    batch_size = 256
    num_epochs = 50
    patience = 10

    lr = 5e-5
    weight_decay = 0.01
    threshold = 0.7
    positive_label_smoothing = 0.1

    device = try_gpu()

    train_iter, val_iter, train_pos_weights = get_soundscapes_dataloader(
        device, batch_size, train_split=0.8, train_samples_per_epoch=28500, validation_total_samples=7200,
        pos_weights_clamp_max=25
    )
    try:
        for net, weights_path in nets:
            if weights_path is not None:
                net.load_weights(weights_path)
            train_model(
                device, net, True, True,
                lr, weight_decay, positive_label_smoothing, threshold, False,
                train_iter, val_iter, train_pos_weights, num_epochs, patience, save_weights=True, # REMINDER: SET TO TRUE WHEN FULLY TRAINING
                save_folder="Soundscapes Training (OneCycleLR)"
            )
    except Exception as e:
        with open("error.txt", "w") as f:
            print('ERROR!')
            f.write(str(e))
    finally:
        # os.system("shutdown /s /t 5")
        pass