import os
import librosa
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
from typing import List


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


class AudioProcessor:

    def __init__(self, src_directory: str, dst_directory: str, treatments: List[str]):
        self.src_directory = src_directory
        self.dst_directory = dst_directory
        self.treatments = treatments
        self.file_extension = ".gz"

    def run(self):
        self.organize_files_by_treatment()
        self.save_spectrograms()
        self.combine_dicts()
        self.open_and_view_json()

    def organize_files_by_treatment(self):
        assert self.src_directory != self.dst_directory, "Source and destination directories cannot be the same."
        for treatment in self.treatments:
            treatment_directory = os.path.join(self.dst_directory, treatment + "_extract")
            os.makedirs(treatment_directory, exist_ok=True)
            for dirpath, dirnames, filenames in os.walk(self.src_directory):
                for filename in filenames:
                    if filename.endswith(self.file_extension) and treatment in dirpath:
                        src_file_path = os.path.join(dirpath, filename)
                        dst_file_path = os.path.join(treatment_directory, filename)
                        shutil.copy2(src_file_path, dst_file_path)
    def save_spectrograms(self, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=30):
        treatments = sorted([dir_name for dir_name in os.listdir(self.src_directory) if os.path.isdir(os.path.join(self.src_directory, dir_name))])

        subfolders_by_treatment = collections.defaultdict(list)
        for treatment in treatments:
            treatment_path = os.path.join(self.src_directory, treatment)
            subfolders = sorted([name for name in os.listdir(treatment_path) if os.path.isdir(os.path.join(treatment_path, name))])
            subfolders_by_treatment[treatment] = subfolders

        max_subfolders = max(len(subfolders) for subfolders in subfolders_by_treatment.values())

        for i in range(max_subfolders):
            for treatment in treatments:
                subfolders = subfolders_by_treatment[treatment]
                if i < len(subfolders):
                    subfolder = subfolders[i]
                    subfolder_path = os.path.join(self.src_directory, treatment, subfolder)

                    for filename in os.listdir(subfolder_path):
                        if filename.endswith(".wav"):
                            self.process_file(filename, subfolder_path, treatment, n_mfcc, n_fft, hop_length, num_segments)

    def process_file(self, filename, subfolder_path, treatment, n_mfcc, n_fft, hop_length, num_segments):
        file_path = os.path.join(subfolder_path, filename)

        dict_data = {
            "light_treatment": [],
            "mfcc": [],
            "labels": [],
            "subfolder_name": [],
            "segment_number": []
        }

        gz_json_file = filename.replace(".wav", ".gz")
        gz_json_path = os.path.join(subfolder_path, gz_json_file)
        if os.path.isfile(gz_json_path):
            print(f"'{gz_json_file}' file already exists in {subfolder_path}. Skipping this file.")
            return

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

            mfcc = librosa.feature.mfcc(y=segment_signal, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc,
                                        hop_length=hop_length)
            mfcc = mfcc.T

            print(f"The file {filename}, section: {s} is being processed...")

            if len(mfcc) == expected_vectors_mfcc:
                dict_data["mfcc"].append(mfcc.tolist())
                dict_data["labels"].append(self.treatments.index(treatment))
                dict_data["light_treatment"].append(treatment)
                dict_data["subfolder_name"].append(self.subfolder)
                dict_data["segment_number"].append(s)

        write_gz_json(dict_data, gz_json_path)

        print(f"The file {filename} has been processed.")

    def combine_dicts(self):
        master_dict = {
            "light_treatment": [],
            "mfcc": [],
            "labels": [],
            "file_name": [],
            "segment_number": []
        }

        for treatment in self.treatments:
            treatment_path = os.path.join(self.src_directory, treatment)
            subfolders = sorted([name for name in os.listdir(treatment_path) if os.path.isdir(os.path.join(treatment_path, name))])

            for subfolder in subfolders:
                subfolder_path = os.path.join(treatment_path, subfolder)

                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".gz"):
                        gz_json_path = os.path.join(subfolder_path, filename)

                        with gzip.GzipFile(gz_json_path, 'r') as gz_file:
                            dict_data = json.loads(gz_file.read().decode('utf-8'))

                            master_dict["light_treatment"].extend(dict_data["light_treatment"])
                            master_dict["mfcc"].extend(dict_data["mfcc"])
                            master_dict["labels"].extend(dict_data["labels"])
                            master_dict["file_name"].extend(dict_data["subfolder_name"])
                            master_dict["segment_number"].extend(dict_data["segment_number"])

        master_file_path = os.path.join(self.dst_directory, "master_file.gz")
        write_gz_json(master_dict, master_file_path)

    def open_and_view_json(self):
        master_file_path = os.path.join(self.dst_directory, "master_file.gz")
        with gzip.GzipFile(master_file_path, 'r') as gz_file:
            master_dict = json.loads(gz_file.read().decode('utf-8'))

        print(master_dict)



