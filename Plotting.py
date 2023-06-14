import os
import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd
def plot_mean_amplitude_envelope(min_size, max_size):
    """
    Plots the mean amplitude envelope for each treatment in a given directory.
    Only processes files within the provided memory size range.

    Args:
        min_size (int): Minimum file size in bytes.
        max_size (int): Maximum file size in bytes.
    """

    # Initialize an empty dictionary to store the concatenated means
    treatment_concatenated_means = {}

    # Open a dialog box for the user to choose a directory
    root = tk.Tk()
    root.withdraw()
    feature_directory = filedialog.askdirectory(title="Please select the feature directory")

    # Open a text file to log problematic files
    with open('problematic_files.txt', 'w') as log_file:

        # Walk through the directory tree
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
                file_segments = np.concatenate(dict_data[os.path.basename(feature_directory)][:29])

                # Check if the size of the concatenated segments is 29*2854
                if file_segments.shape[0] != 29*2584:
                    log_file.write(f"Problematic file: {file_path} | Size: {file_segments.shape[0]}\n")
                    continue

                # Append this array to the current treatment's list
                treatment_concatenated_means[treatment].append(file_segments)
                print(f"Shape of file_segments for {treatment}: {file_segments.shape}")

                print(f"Number of files processed for {treatment}: {len(treatment_concatenated_means[treatment])}")

        # Compute means for each treatment and plot them
        for treatment, concatenated_segments_list in treatment_concatenated_means.items():
            print(f"\nCalculating mean for {treatment}.")

            # Compute the mean across the concatenated segments for the current treatment
            mean_features = np.mean(concatenated_segments_list, axis=0)
            print(f"Shape of mean_features for {treatment}: {mean_features.shape}")
            # sample_index = np.linspace(0, len(mean_features)-1, len(mean_features))
            mean_features_series = pd.Series(mean_features)

            # Apply the rolling mean with a window size of 10
            smooth_mean = mean_features_series.rolling(window=100).mean()

            # The first few values of smooth_mean will be NaN (due to the window size), so you might want to handle them before proceeding.
            # smooth_mean = smooth_mean.dropna()

            # The corresponding sample_index should also be adjusted to match the length of smooth_mean
            sample_index = np.linspace(0, len(smooth_mean) - 1, len(smooth_mean))


            # Plot the mean amplitude envelope
            plt.plot(sample_index, smooth_mean, label=treatment)

        # Configure the plot
        plt.xlabel('Sample_index')
        plt.ylabel('Mean Amplitude Envelope')
        plt.title('Mean Amplitude Envelope per Treatment')
        plt.legend()

        # Display the plot
        plt.show()



# plot_mean_zero_crossing_rate(100000, 200000)
plot_mean_amplitude_envelope(80000, 200000)