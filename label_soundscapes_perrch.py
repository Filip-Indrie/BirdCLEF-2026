import pandas as pd
import os
import time
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import tqdm

from load_utils import TARGET_FRAMES, TARGET_SECONDS, TARGET_SAMPLE_RATE

dataset_folder = ".."

def format_timestamp(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

if __name__ == "__main__":
    perch_labels_df = pd.read_csv(f"{dataset_folder}/perch_output_labels.csv")
    perch_classes = perch_labels_df['ebird2021'].tolist()

    ebird_taxonomy_df = pd.read_csv(f"{dataset_folder}/eBird_Taxonomy_v2021.csv")
    ebird_to_sci_name = pd.Series(
        ebird_taxonomy_df.SCI_NAME.values,
        index=ebird_taxonomy_df.SPECIES_CODE
    ).to_dict()

    taxonomy_df = pd.read_csv(f"{dataset_folder}/taxonomy.csv")
    sci_name_to_label = pd.Series(
        taxonomy_df.primary_label.values,
        index=taxonomy_df.scientific_name
    ).to_dict()

    model_url = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/4"
    perch_model = hub.load(model_url)

    unlabeled_dir = f"{dataset_folder}/train_soundscapes"
    file_list = [f for f in os.listdir(unlabeled_dir) if f.endswith(".ogg")]

    final_results = []

    for filename in tqdm.tqdm(file_list, desc="Soundscapes"):
        file_path = os.path.join(unlabeled_dir, filename)

        audio, sr = sf.read(file_path, dtype="float32")
        if sr != TARGET_SAMPLE_RATE: continue

        target_total_frames = TARGET_FRAMES * 12 # TARGET_FRAMES = TARGET_SR (32k) * TARGET_SECONDS (5)
        if len(audio) < target_total_frames:
            audio = np.pad(audio, (0, target_total_frames - len(audio)))
        else:
            audio = audio[:target_total_frames]

        audio_batch = audio.reshape(-1, TARGET_FRAMES)
        audio_tensor = tf.convert_to_tensor(audio_batch, dtype=tf.float32)

        logits, embeddings = perch_model.infer_tf(audio_tensor)
        probabilities = tf.nn.sigmoid(logits).numpy()

        for chunk in range(12):
            bin_start_sec = chunk * TARGET_SECONDS
            bin_start_str = format_timestamp(bin_start_sec)
            bin_end_str = format_timestamp(bin_start_sec + TARGET_SECONDS)

            chunk_probs = probabilities[chunk]

            active_indices = np.where(chunk_probs > 0.5)[0]
            valid_labels = set()

            for idx in active_indices:
                ebird_code = perch_classes[idx]
                if ebird_code in ebird_to_sci_name:
                    sci_name = ebird_to_sci_name[ebird_code]
                    if sci_name in sci_name_to_label:
                        label = str(sci_name_to_label[sci_name])
                        valid_labels.add(label)

            if len(valid_labels) == 0:
                primary_label = "nocall"
            else:
                primary_label = ";".join(sorted(list(valid_labels)))

            final_results.append({
                "filename": filename,
                "start": bin_start_str,
                "end": bin_end_str,
                "primary_label": primary_label
            })

    df = pd.DataFrame(final_results, columns=["filename", "start", "end", "primary_label"])
    df = df.sort_values(by=["filename", "start"], ascending=[True, True]).reset_index(drop=True)
    df.to_csv(f"{dataset_folder}/perch_soundscapes_labels.csv", index=False)

    # os.system("shutdown /s /t 5")