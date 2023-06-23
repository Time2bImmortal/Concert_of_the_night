import os
import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd
FEATURES_TO_PLOT = {
    'ae': (80000, 200000),
    "rms": (600000, 800000),
    "zcr": (100000, 200000),
    "bw": (600000, 800000),
    "sc": (600000, 800000),
    "ber": (400000, 750000)
}

def plot_mean_feature_treatment(directory: str, data_dict: dict):
    """
    Plots the mean feature for each treatment in a given directory.
    Only processes files within the provided memory size range.

    Args:
        directory (str): The directory where the function will look for the files.
        data_dict (dict): The dictionary containing the features to be plotted
                          along with their min and max sizes.
    """

    # Walk through the directory tree
    for feature_name, (min_size, max_size) in data_dict.items():
        # Initialize an empty dictionary to store the concatenated means
        treatment_concatenated_means = {}

        feature_directory = os.path.join(directory, feature_name)
        for dirpath, dirs, files in os.walk(feature_directory):
            print(f"\nProcessing directory: {dirpath}")

            for file in files:
                file_path = os.path.join(dirpath, file)

                # Check if file size is within the given range
                file_size = os.path.getsize(file_path)
                if file_size < min_size or file_size > max_size:
                    continue

                print(f"Processing file: {file}")

                # Open the gzipped JSON file and load the data
                with gzip.GzipFile(file_path, 'r') as fin:
                    dict_data = json.loads(fin.read().decode('utf-8'))

                # Check if the current treatment is already in the dictionary, if not initialize it
                treatment = os.path.basename(dirpath)
                if treatment not in treatment_concatenated_means:
                    treatment_concatenated_means[treatment] = []

                # Concatenate the segments in the current file
                file_segments = np.concatenate(dict_data[feature_name][:29])

                # Append this array to the current treatment's list
                treatment_concatenated_means[treatment].append(file_segments)
                print(f"Shape of file_segments for {treatment}: {file_segments.shape}")

                print(f"Number of files processed for {treatment}: {len(treatment_concatenated_means[treatment])}")

        # Compute means for each treatment and plot them
        for treatment, concatenated_segments_list in treatment_concatenated_means.items():
            print(f"\nCalculating mean for {treatment}.")
            # This is just before you calculate the mean
            try:
                # Filter out elements that do not have the expected shape
                expected_shape = (29, 2584)
                concatenated_segments_list = [segment for segment in concatenated_segments_list if
                                              np.shape(segment) == expected_shape]

                # Check the shape of the filtered concatenated_segments_list
                print("concatenated_segments_list shape after filtering:", np.shape(concatenated_segments_list))

                # Compute the mean across the concatenated segments for the current treatment
                mean_features = np.mean(concatenated_segments_list, axis=0)
                print(f"Shape of mean_features for {treatment}: {mean_features.shape}")
            except ValueError as e:
                print(f"An error occurred while processing treatment {treatment}: {e}")
                continue

            # # Compute the mean across the concatenated segments for the current treatment
            # mean_features = np.mean(concatenated_segments_list, axis=0)
            # mean_features = mean_features.flatten()
            #
            # print(f"Shape of mean_features for {treatment}: {mean_features.shape}")
            # mean_features_series = pd.Series(mean_features)
            mean_features_series = pd.Series(mean_features.flatten())
            smooth_mean = mean_features_series.rolling(window=100).mean()

            # Apply the rolling mean with a window size of 10
            # smooth_mean = mean_features_series.rolling(window=100).mean()

            # The corresponding sample_index should also be adjusted to match the length of smooth_mean
            sample_index = np.linspace(0, len(smooth_mean) - 1, len(smooth_mean))

            # Plot the mean amplitude envelope
            plt.plot(sample_index, smooth_mean, label=treatment)

        # Configure the plot
        plt.xlabel('Sample_index')
        plt.ylabel(f'Mean {feature_name}')
        plt.title(f'Mean {feature_name} per Treatment')
        plt.legend()
        plot_dir = os.path.join(folder_path, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{feature_name}_plot.png"))

        # Display the plot
        plt.show()




if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    plot_mean_feature_treatment(folder_path, FEATURES_TO_PLOT)