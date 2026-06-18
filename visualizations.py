from matplotlib import pyplot as plt
import json
import re


def parse_training_stats(input_filepath, output_filepath):
    data = {
        "num_params": 0,
        "training_time": 0.0,
        "train_loss": [],
        "train_f1": [],
        "train_precision": [],
        "train_recall": [],
        "val_loss": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": []
    }

    with open(input_filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    params_match = re.search(r"Total params:\s*([\d,]+)", content)
    if params_match:
        data["num_params"] = int(params_match.group(1).replace(",", ""))

    time_match = re.search(r"Training Time:\s*([\d.]+)s", content)
    if time_match:
        data["training_time"] = float(time_match.group(1))

    train_pattern = r"Train stats:\s*Loss:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)\s*\|\s*Precision:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)"
    val_pattern = r"Validation stats:\s*Loss:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)\s*\|\s*Precision:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)"

    train_matches = re.findall(train_pattern, content)
    val_matches = re.findall(val_pattern, content)

    for match in train_matches:
        data["train_loss"].append(float(match[0]))
        data["train_f1"].append(float(match[1]))
        data["train_precision"].append(float(match[2]))
        data["train_recall"].append(float(match[3]))

    for match in val_matches:
        data["val_loss"].append(float(match[0]))
        data["val_f1"].append(float(match[1]))
        data["val_precision"].append(float(match[2]))
        data["val_recall"].append(float(match[3]))

    with open(output_filepath, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

def load_model_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_stages(models_data):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model F1 Score Comparison Across Training Stages', fontsize=18, fontweight='bold', y=0.95)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    models = list(models_data.keys())

    def plot_metric(ax, stage, metric, title):
        for i, model in enumerate(models):
            model_data = models_data[model][stage][metric]
            epochs = range(1, len(model_data) + 1)
            ax.plot(epochs, model_data, label=model, color=colors[i % len(colors)], linewidth=2)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=11)

    plot_metric(axs[0, 0], stage='stage1', metric='train_f1', title='Stage 1: Training F1')
    plot_metric(axs[0, 1], stage='stage1', metric='val_f1', title='Stage 1: Validation F1')
    plot_metric(axs[1, 0], stage='stage2', metric='train_f1', title='Stage 2: Training F1')
    plot_metric(axs[1, 1], stage='stage2', metric='val_f1', title='Stage 2: Validation F1')

    # plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.savefig("model_comparison_grid.png", dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    # file_list = [("C:\\Users\deepo\Desktop\Poli Stuff\_Extras\Side Projects\BirdCLEF+ 2026\Code\Measurements\ResNet18\Focal loss, Label smoothing, Noise Introduction\Single Bird Training (Focal loss, Label smoothing, Noise Introduction)\stats.txt",
    #               "C:\\Users\deepo\Desktop\Poli Stuff\_Extras\Side Projects\BirdCLEF+ 2026\Code\Measurements\ResNet18\Focal loss, Label smoothing, Noise Introduction\Single Bird Training (Focal loss, Label smoothing, Noise Introduction)\\all_measurements.json"),
    #              ("C:\\Users\deepo\Desktop\Poli Stuff\_Extras\Side Projects\BirdCLEF+ 2026\Code\Measurements\ResNet18\POS_WEIGHT_CLAMP_10\Soundscapes ALL Fine-Tune\stats.txt",
    #               "C:\\Users\deepo\Desktop\Poli Stuff\_Extras\Side Projects\BirdCLEF+ 2026\Code\Measurements\ResNet18\POS_WEIGHT_CLAMP_10\Soundscapes ALL Fine-Tune\\all_measurements.json")]
    #
    # for input_file, output_file in file_list:
    #     parse_training_stats(input_file, output_file)

    data = {
        "ResNet18": {
            "stage1": load_model_data("./Measurements/ResNet18/Focal loss, Label smoothing, Noise Introduction/Single Bird Training (Focal loss, Label smoothing, Noise Introduction)/all_measurements.json"),
            "stage2": load_model_data("./Measurements/ResNet18/POS_WEIGHT_CLAMP_10/Soundscapes ALL Fine-Tune/all_measurements.json")
        },
        "EfficientNetB1": {
            "stage1": load_model_data("./Measurements/EfficientNetB1/Training (240x240) 50 epochs/Single Birds/all_measurements.json"),
            "stage2": load_model_data("./Measurements/EfficientNetB1/Training (240x240) 50 epochs/Soundscapes All Fine-Tune/all_measurements.json")
        },
        "InceptionV3": {
            "stage1": load_model_data("./Measurements/InceptionV3/Training (299x299)/Single Birds/all_measurements.json"),
            "stage2": load_model_data("./Measurements/InceptionV3/Training (299x299, less augmented)/Soundscapes ALL Fine-Tune/all_measurements.json")
        }
    }

    plot_training_stages(data)

    pass