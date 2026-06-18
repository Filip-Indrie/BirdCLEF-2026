import os
import torch
import pandas as pd
import random
from dotenv import load_dotenv

from spectrogram_models import InceptionV3
from train_utils import try_gpu, get_spectrogram_transform, wave_to_spectrogram, init_spectrogram
from load_utils import get_single_bird_dataloader, get_soundscapes_dataloader

load_dotenv()
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", 32000))

df = pd.read_csv("../taxonomy.csv")
classes = df['primary_label'].unique().tolist()
IDX_TO_BIRD = {str(i): bird for i, bird in enumerate(classes)}

def demo(threshold, device, model, mel_transform, spectrogram_resize, batches):
    wave, true_labels = random.choice(batches)
    wave = wave.to(device)
    true_labels = true_labels.squeeze(0)

    # Run Inference
    with torch.no_grad():
        spectrogram = wave_to_spectrogram(wave, mel_transform)
        spectrogram = spectrogram_resize(spectrogram)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(spectrogram)
            probs = torch.sigmoid(logits).squeeze(0)

    actual_birds = []
    for idx, is_present in enumerate(true_labels):
        if is_present == 1.0:
            actual_birds.append(IDX_TO_BIRD[str(idx)])
    if not actual_birds: actual_birds.append("nocall")

    predicted_birds = []
    for idx, prob in enumerate(probs):
        if prob.item() > threshold:
            predicted_birds.append((IDX_TO_BIRD[str(idx)], prob.item()))
    predicted_birds.sort(key=lambda x: x[1], reverse=True)

    print("GROUND TRUTH:")
    for bird in actual_birds:
        print(f"{bird}")

    print("-" * 50)

    print("MODEL PREDICTION:")
    if not predicted_birds:
        print("nocall")
    else:
        for bird, prob in predicted_birds:
            print(f"{bird} ({prob * 100:.2f}% confidence)")
    print("=" * 50 + "\n")

def demo_single_birds(device, model, mel_transform, spectrogram_resize):
    net.load_weights("./Measurements/InceptionV3/Training (299x299)/Single Birds/weights.pth")
    net.eval()

    _, val_iter, _ = get_single_bird_dataloader(_device, batch_size=1, train_split=0.8)

    all_batches = list(val_iter)

    for i in range(10):
        demo(0.7, device, model, mel_transform, spectrogram_resize, all_batches)

def demo_soundscapes(device, model, mel_transform, spectrogram_resize):
    net.load_weights("./Measurements/InceptionV3/Training (299x299, less augmented)/Soundscapes ALL Fine-Tune/weights.pth")
    net.eval()

    _, val_iter, _ = get_soundscapes_dataloader(
        _device, batch_size=1, train_split=0.8,
        train_samples_per_epoch=1, validation_total_samples=100
    )

    all_batches = list(val_iter)

    for i in range(10):
        demo(0.45, device, model, mel_transform, spectrogram_resize, all_batches)

if __name__ == '__main__':
    _device = try_gpu()
    net = InceptionV3()
    net.to(_device)

    _mel_transform = init_spectrogram(_device)
    _spectrogram_resize = get_spectrogram_transform(299, 299).to(_device)

    demo_single_birds(_device, net, _mel_transform, _spectrogram_resize)

    print('\n')
    print("=-" * 30)
    print("Soundscapes:")

    demo_soundscapes(_device, net, _mel_transform, _spectrogram_resize)