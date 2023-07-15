import os
import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.patches as mpatches
FEATURES_TO_PLOT = {
    'ae': (70000, 250000),
    "rms": (500000, 750000),
    "zcr": (100000, 200000),
    "bw": (500000, 800000),
    "sc": (500000, 800000),
    "ber": (400000, 750000)
}


def plot_mean_feature_treatment(directory: str, data_dict: dict, per_folder: bool = False):
    treatment_color = {
        '2lux': 'blue',
        '5lux': 'orange',
        'LD': 'red',
        'LL': 'green'
    }
    plots_directory = os.path.join(directory, 'Plots')
    os.makedirs(plots_directory, exist_ok=True)
    for feature_name, (min_size, max_size) in data_dict.items():
        print(f"{feature_name} is being plotted")
        feature_directory = os.path.join(directory, feature_name)
        if per_folder:
            # process_treatments(treatment_color, feature_directory, feature_name, min_size, max_size)
            process_subfolders(treatment_color, feature_directory, feature_name, min_size, max_size)
        else:
            process_treatments(treatment_color, feature_directory, feature_name, min_size, max_size)
        legend_patches = [mpatches.Patch(color=c, label=l) for l, c in treatment_color.items()]
        plt.legend(handles=legend_patches)
        if per_folder:
            plt.savefig(f'{plots_directory}/{feature_name}_plot_per_folders.png')
        else:
            plt.savefig(f'{plots_directory}/{feature_name}_plot.png')

        print(f"{feature_name} has been plotted")

        # Clear the current figure's content
        plt.clf()

def process_treatments(treatment_color, feature_directory, feature_name, min_size, max_size):
    treatment_means_list = []
    treatment_list = [t for t in os.listdir(feature_directory) if os.path.isdir(os.path.join(feature_directory, t))]

    for treatment in treatment_list:
        treatment_directory = os.path.join(feature_directory, treatment)

        treatment_files = []
        for root, dirs, files in os.walk(treatment_directory):
            # Check if there are gzip files directly in the treatment directory
            treatment_files.extend([
                os.path.join(root, file)
                for file in files
                if file.endswith('.gz')
            ])

            # Check if there are subfolders containing gzip files
            for subdir in dirs:
                subfolder_path = os.path.join(root, subdir)
                subfolder_files = [
                    os.path.join(subfolder_path, file)
                    for file in os.listdir(subfolder_path)
                    if file.endswith('.gz')
                ]
                treatment_files.extend(subfolder_files)

        treatment_means = calculate_means_from_files(treatment_files, feature_name, min_size, max_size)
        treatment_means_list.append((treatment, treatment_means))
    for treatment, treatment_means in treatment_means_list:
        plot_means(treatment_means, treatment_color[treatment], feature_name, alpha=1.0)


def calculate_means_from_files(files, feature_name, min_size, max_size):

    files_concatenated = []
    for file in files:
        if min_size < os.path.getsize(file) < max_size:
            with gzip.GzipFile(file, 'r') as fin:
                dict_data = json.loads(fin.read().decode('utf-8'))

            file_segments = np.concatenate(dict_data[feature_name][:29], axis=1)  # Concatenation on axis=1
            if file_segments.shape == (1, 74936):  # Condition to consider arrays with shape (1, 74936)
                files_concatenated.append(file_segments)
    if not files_concatenated:
        print("Warning: No valid means were calculated for the given files.")
        return None
    return np.mean(files_concatenated, axis=0)


def process_subfolders(treatment_color, feature_directory, feature_name, min_size, max_size):
    treatment_list = [t for t in os.listdir(feature_directory) if os.path.isdir(os.path.join(feature_directory, t))]
    for treatment in treatment_list:
        treatment_directory = os.path.join(feature_directory, treatment)
        # Assuming that all subdirectories under the treatment are subfolders.
        subfolder_mean = []
        for subfolder in os.listdir(treatment_directory):
            subfolder_path = os.path.join(treatment_directory, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # Get all files from the subfolder.
            subfolder_files = [os.path.join(subfolder_path, file) for file in os.listdir(subfolder_path) if
                               os.path.isfile(os.path.join(subfolder_path, file))]

            subfolder_mean.append(calculate_means_from_files(subfolder_files, feature_name, min_size, max_size))
        for mean in subfolder_mean:
            plot_means(mean, treatment_color[treatment], feature_name, alpha=1.0, linewidth=0.4)


def plot_means(mean_values, color, feature_name, alpha=1.0, linewidth=2.0):
    if mean_values is None:
        return
    mean_features_series = pd.Series(mean_values.flatten())
    smooth_mean = mean_features_series.rolling(window=250).mean()
    sample_index = np.linspace(0, len(smooth_mean) - 1, len(smooth_mean))
    plt.plot(sample_index, smooth_mean, color=color, alpha=alpha, linewidth=linewidth)
    plt.xlabel('Sample_index')
    plt.ylabel(f'Mean {feature_name}')
    plt.title(f'Mean {feature_name} per Treatment')




if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    plot_mean_feature_treatment(folder_path, FEATURES_TO_PLOT, per_folder=True)