import os
import librosa, librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog
import math
import gzip
import collections
import shutil


def organize_files_by_treatment(src_directory, dst_directory, treatments, file_extension=".gz"):
    """
    Organize files by treatments.

    Parameters:
    - src_directory: Source directory where the .gz files are located.
    - dst_directory: Destination directory where the files will be organized.
    - treatments: List of treatments.
    - file_extension: File extension to be copied. Default is ".gz".
    """
    # Make sure that source and destination directories are different
    assert src_directory != dst_directory, "Source and destination directories cannot be the same."

    # Iterate over each treatment
    for treatment in treatments:
        # Create a new directory for each treatment, adding '_extract' to the name
        treatment_directory = os.path.join(dst_directory, treatment + "_extract")
        os.makedirs(treatment_directory, exist_ok=True)

        # Go through all subdirectories, files in source directory
        for dirpath, dirnames, filenames in os.walk(src_directory):
            # For each file, if it's a .gz file and if the file belongs to the current treatment
            for filename in filenames:
                if filename.endswith(file_extension) and treatment in dirpath:
                    # Form the full file path in source and destination
                    src_file_path = os.path.join(dirpath, filename)
                    dst_file_path = os.path.join(treatment_directory, filename)

                    # Copy the file to the new directory
                    shutil.copy2(src_file_path, dst_file_path)


def save_spectrograms(dataset_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=30):
    treatments = sorted([dir_name for dir_name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, dir_name))])

    subfolders_by_treatment = collections.defaultdict(list)
    for treatment in treatments:
        treatment_path = os.path.join(dataset_path, treatment)
        subfolders = sorted([name for name in os.listdir(treatment_path) if os.path.isdir(os.path.join(treatment_path, name))])
        subfolders_by_treatment[treatment] = subfolders

    max_subfolders = max(len(subfolders) for subfolders in subfolders_by_treatment.values())

    for i in range(max_subfolders):
        for treatment in treatments:
            subfolders = subfolders_by_treatment[treatment]
            if i < len(subfolders):
                subfolder = subfolders[i]
                subfolder_path = os.path.join(dataset_path, treatment, subfolder)

                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".wav"):
                        file_path = os.path.join(subfolder_path, filename)

                        dict_data = {
                            "light_treatment": [],
                            "mfcc": [],
                            # "spectrogram": [],
                            "labels": [],
                            "subfolder_name": [],
                            "segment_number": []
                        }

                        gz_json_file = filename.replace(".wav", ".gz")  # Changed file extension
                        gz_json_path = os.path.join(subfolder_path, gz_json_file)
                        if os.path.isfile(gz_json_path):
                            print(f"'{gz_json_file}' file already exists in {subfolder_path}. Skipping this file.")
                            continue

                        signal, sr = librosa.load(file_path)
                        duration = librosa.get_duration(y=signal, sr=sr)
                        samples_per_track = sr*duration
                        num_samples_per_segment = samples_per_track / num_segments
                        expected_vectors_mfcc = math.ceil(num_samples_per_segment / hop_length)

                        for s in range(num_segments):
                            start_sample = int(num_samples_per_segment * s)
                            end_sample = int(start_sample + num_samples_per_segment)
                            segment_signal = signal[start_sample:min(end_sample, len(signal))]

                            if len(segment_signal) == 0:
                                print(f"Empty segment signal at segment {s} of file {file_path}. Skipping this segment.")
                                continue

                            # stft = librosa.core.stft(segment_signal, hop_length=hop_length, n_fft=n_fft)
                            # spectrogram = np.abs(stft)
                            mfcc = librosa.feature.mfcc(y=segment_signal, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                            mfcc = mfcc.T

                            print(f"The file {filename}, section: {s} is being processed...")

                            if len(mfcc) == expected_vectors_mfcc:
                                dict_data["mfcc"].append(mfcc.tolist())
                                dict_data["labels"].append(treatments.index(treatment))
                                # dict_data["spectrogram"].append(spectrogram.tolist())
                                dict_data["light_treatment"].append(treatment)
                                dict_data["subfolder_name"].append(subfolder)
                                dict_data["segment_number"].append(s)

                        # gz_json_path = json_path.replace(".json", ".gz")
                        write_gz_json(dict_data, gz_json_path)

                        print(f"The file {filename} has been processed.")


def combine_dicts(dataset_path):
    master_dict = {
        "light_treatment": [],
        "mfcc": [],
        "spectrogram": [],
        "labels": [],
        "file_name": [],
        "segment_number": []
    }

    for root, dirs, files in os.walk(dataset_path):
        if 'data.json' in files:
            with open(os.path.join(root, 'data.json'), 'r') as fp:
                data_dict = json.load(fp)
                for key in master_dict.keys():
                    master_dict[key].extend(data_dict[key])

    with open(os.path.join(dataset_path, 'master_data.json'), 'w') as fp:
        json.dump(master_dict, fp, indent=5)

    print(f"'master_data.json' saved in {dataset_path}.")


def open_and_view_json():
    root = tk.Tk()
    root.withdraw()  # to hide the small tk window

    # open file dialog to choose a file
    file_path = filedialog.askopenfilename(filetypes=[('GZ Files', '*.gz')])

    fig, ax = plt.subplots()
    i = 0  # define i here

    if file_path:  # if a file was chosen
        with gzip.open(file_path, 'rt') as fp:  # 'rt' mode to open as text file
            data_dict = json.load(fp)

        # MFCC data extraction
        mfcc_data = np.array(data_dict["mfcc"])

        # Initial plot
        im = ax.imshow(mfcc_data[i].T, aspect='auto', cmap='hot_r', origin='lower')
        fig.colorbar(im)

        # Respond to a key press
        def on_key(event):
            nonlocal i  # reference the i defined in the enclosing scope
            if event.key == 'right':
                i = (i + 1) % len(mfcc_data)
                print(f"Segment n_{i}/30")
            elif event.key == 'left':
                i = (i - 1) % len(mfcc_data)
                print(f"Segment n_{i}/30")

            # Update the image data
            im.set_data(mfcc_data[i].T)
            # Redraw the figure
            fig.canvas.draw()

        # Connect the event to the function
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    else:
        print("No file chosen.")


def get_samplerate(audio_file_path):
    data, samplerate = sf.read(audio_file_path)
    print(f'The file has a samplerate of: {samplerate}')
    return samplerate


def display_waveform(signal, sr):
    plt.figure()
    librosa.display.waveshow(signal, sr=sr)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()


def write_gz_json(json_obj, filename):
    json_str = json.dumps(json_obj) + "\n"
    json_bytes = json_str.encode('utf-8')

    with gzip.GzipFile(filename, 'w') as fout:
        fout.write(json_bytes)


def save_and_compare_audio(filename):
    # Load the audio file
    signal, sr = librosa.load(filename)

    # Save the audio data to a json file
    json_filename = filename.replace('.wav', '.json')
    with open(json_filename, 'w') as json_file:
        json.dump(signal.tolist(), json_file)

    # Save the audio data to a gzipped json file
    gz_filename = filename.replace('.wav', '.gz')
    write_gz_json(signal.tolist(), gz_filename)

    # Load the json data
    with open(json_filename, 'r') as json_file:
        json_data = np.array(json.load(json_file))

    # Load the gzipped json data
    with gzip.GzipFile(gz_filename, 'r') as gz_file:
        gz_data = np.array(json.loads(gz_file.read().decode('utf-8')))

    # Compare the two data arrays
    if np.array_equal(json_data, gz_data):
        print("The two files contain identical data.")
    else:
        print("The two files do not contain identical data.")

# Choose an audio file
# print("First input source")
root = tk.Tk()
root.withdraw()
# dataset_path = filedialog.askdirectory()
# print("Then folder")
# goal = filedialog.askdirectory()
filename = filedialog.askopenfilename()
# save_spectrograms(dataset_path)
# organize_files_by_treatment(dataset_path,goal, ["2lux", "5lux", "LL", "LD"])
# open_and_view_json()

save_and_compare_audio(filename)