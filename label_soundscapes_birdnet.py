import pandas as pd
import os
import time
import tqdm
import math
import soundfile
import concurrent.futures
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

dataset_folder = ".."

def format_timestamp(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

worker_analyzer = None
worker_taxonomy = dict()

def init_worker(taxonomy_dict):
    global worker_analyzer, worker_taxonomy
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    worker_analyzer = Analyzer()
    worker_taxonomy = taxonomy_dict

def process_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        if round(soundfile.info(file_path).duration) != 60: return []

        recording = Recording(
            worker_analyzer,
            file_path
        )

        recording.analyze()

        results = {}

        for detection in recording.detections:
            if detection['confidence'] > 0.5:
                sci_name = detection['scientific_name']
                if sci_name in worker_taxonomy:
                    numeric_id = str(worker_taxonomy[sci_name])

                    start_sec = detection['start_time']
                    end_sec = detection['end_time']

                    start_frame = int(math.floor(start_sec / 5.0) * 5)
                    end_frame = int(math.floor(end_sec / 5.0) * 5)

                    for frame_sec in range(start_frame, end_frame + 5, 5):
                        bin_start_str = format_timestamp(frame_sec)
                        bin_end_str = format_timestamp(frame_sec + 5)

                        frame_key = (file_name, bin_start_str, bin_end_str)
                        if frame_key not in results:
                            results[frame_key] = set()

                        results[frame_key].add(numeric_id)
                else:
                    print(f"DROPPED: '{sci_name}' (Conf: {detection['confidence']:.2f}) - Not in taxonomy.csv")

        for frame_sec in range(0, 60, 5):
            frame_start_str = format_timestamp(frame_sec)
            frame_end_str = format_timestamp(frame_sec + 5)
            chunk_key = (file_name, frame_start_str, frame_end_str)
            if chunk_key not in results or len(results[chunk_key]) == 0:
                results[chunk_key] = {"nocall"}

        formatted_results = []
        for (file, start_str, end_str), label_set in results.items():
            primary_label = ";".join(sorted(list(label_set)))
            formatted_results.append({
                "filename": file_name,
                "start": start_str,
                "end": end_str,
                "primary_label": primary_label
            })

        return formatted_results
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

if __name__ == "__main__":
    taxonomy_df = pd.read_csv(f"{dataset_folder}/taxonomy.csv")
    sci_name_to_label = pd.Series(
        taxonomy_df.primary_label.values,
        index=taxonomy_df.scientific_name
    ).to_dict()

    unlabeled_dir = "../train_soundscapes"
    file_list = [f for f in os.listdir(unlabeled_dir) if f.endswith(".ogg")]

    final_results = []

    max_cores = max(1, os.cpu_count() - 2)
    print(f"Igniting {max_cores} parallel CPU workers...")

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_cores,
            initializer=init_worker,
            initargs=(sci_name_to_label,)
    ) as executor:
        futures = {executor.submit(process_file, f'{unlabeled_dir}/{filename}'): filename for filename in file_list}

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(file_list), desc="Labeling Soundscapes"):
            file_results = future.result()
            final_results.extend(file_results)

    # Export
    df = pd.DataFrame(final_results)
    df = df.sort_values(by=["filename", "start"]).reset_index(drop=True)
    df.to_csv(f"{dataset_folder}/birdnet_pseudo_labels.csv", index=False)
    print("\nFinished generating mapped pseudo-labels!")