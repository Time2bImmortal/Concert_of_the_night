import os
import librosa
import matplotlib.pyplot as plt
from matplotlib import animation
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
FEATURE_ABBREVIATIONS = {
    "mfcc": "mfcc",
    "amplitude_envelope": "ae",
    "rms": "rms",
    "zero_crossing_rate": "zcr"
}


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


class AudioExplorer:
    def __init__(self, filename, full=False):
        self.signal, self.sr = librosa.load(filename)
        self.full = full
        self.duration = 60 if not self.full else len(self.signal) / self.sr
        self.current_time = 0
        self.fig, self.ax = plt.subplots()

    def display_waveform(self):
        self.ax.clear()
        if self.full:
            librosa.display.waveshow(self.signal, sr=self.sr, ax=self.ax)
            self.ax.set(title='Full Waveform', xlabel='Time (s)', ylabel='Amplitude')
        else:
            start_sample, end_sample = librosa.time_to_samples([self.current_time, self.current_time + self.duration], sr=self.sr)
            segment = self.signal[start_sample:end_sample]
            times = np.linspace(self.current_time, self.current_time + self.duration, num=segment.shape[0])
            self.ax.plot(times, segment)
            self.ax.set(title='Waveform', xlabel='Time (s)', ylabel='Amplitude')

        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right' and (self.current_time + 2 * self.duration) * self.sr < len(self.signal):
            self.current_time += self.duration
        elif event.key == 'left' and self.current_time >= self.duration:
            self.current_time -= self.duration

        self.display_waveform()


def process_audio_interactive(full=False):
    # Prompt the user to choose a file
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename()

    if filename:
        # Display the memory size of the file
        print(f'The file size is: {os.path.getsize(filename)} bytes')

        explorer = AudioExplorer(filename, full)
        explorer.fig.canvas.mpl_connect('key_press_event', explorer.on_key)
        explorer.display_waveform()
        plt.show()
    else:
        print('No file selected.')


class AudioProcessor:
    def __init__(self, feature: str,
                 n_mfcc=13, n_fft=2048, hop_length=512, num_segments=30):
        self.feature = feature  # Feature to extract from audio files
        self.file_extension = ".gz"
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_segments = num_segments
        self.treatments = []
        self.features_dir = None

    def choose_file(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select File")
        root.destroy()
        return file_path
    def single_file(self):
        file = self.choose_file()
        self.process_file(file)
    def create_feature_directory(self, directory):
        feature_directory = os.path.join(os.path.dirname(directory), self.feature)
        os.makedirs(feature_directory, exist_ok=True)
        return feature_directory

    def choose_src_directory(self):
        root = tk.Tk()
        root.withdraw()
        src_directory = filedialog.askdirectory(title="Select Source Directory")
        root.destroy()
        return src_directory

    def create_treatment_directories(self, src_directory):
        treatments = os.listdir(src_directory)
        self.treatments = treatments  # Fill treatments list
        for treatment in treatments:
            treatment_dir = os.path.join(self.features_dir, treatment)
            os.makedirs(treatment_dir, exist_ok=True)

    def run(self):
        src_directory = self.choose_src_directory()
        self.features_dir = self.create_feature_directory(src_directory)
        self.create_treatment_directories(src_directory)
        self.process_file(self.choose_file())


    def process_file(self, file_path):
        filename = os.path.basename(file_path)
        subfolder_path = os.path.dirname(file_path)
        subfolder = os.path.basename(subfolder_path)

        treatment_path = os.path.dirname(subfolder_path)
        treatment = os.path.basename(treatment_path)
        dict_data = {
            "path" : file_path,
            "subfolder_name": subfolder,
            "filename": filename,
            self.feature: [],
            "labels": [],
            "segment_number": []
        }

        gz_json_file = filename.replace(".wav", f"_{self.feature}.gz")
        treatment_dir = os.path.join(self.features_dir, treatment)  # Get the corresponding treatment directory
        os.makedirs(treatment_dir, exist_ok=True)  # Create treatment directory if it doesn't exist
        gz_json_path = os.path.join(treatment_dir, gz_json_file)
        if os.path.isfile(gz_json_path):
            print(f"'{gz_json_file}' file already exists in {treatment_dir}. Skipping this file.")
            return

        signal, sr = librosa.load(file_path)
        duration = librosa.get_duration(y=signal, sr=sr)
        samples_per_track = sr * duration
        num_samples_per_segment = samples_per_track / self.num_segments
        expected_vectors_mfcc = math.ceil(num_samples_per_segment / self.hop_length)

        for s in range(self.num_segments):
            start_sample = int(num_samples_per_segment * s)
            end_sample = int(start_sample + num_samples_per_segment)
            segment_signal = signal[start_sample:end_sample]

            if len(segment_signal) == 0:
                print(f"Empty segment signal at segment {s} of file {file_path}. Skipping this segment.")
                continue

            feature_vectors = self.extract_feature(segment_signal, sr)
            feature_vectors = feature_vectors.T

            print(f"The file {filename}, section: {s} is being processed...")

            if len(feature_vectors) == expected_vectors_mfcc:
                dict_data[self.feature].append(feature_vectors.tolist())
                dict_data["labels"].append(self.treatments.index(treatment))
                dict_data["segment_number"].append(s)

        self.write_gz_json(dict_data, gz_json_path)

        print(f"The file {filename} has been processed.")

    def extract_feature(self, signal, sr):
        if self.feature == "mfcc":
            return librosa.feature.mfcc(y=signal, sr=sr)
        elif self.feature == "amplitude_envelope":
            return librosa.amplitude_to_db(librosa.feature.rmse(signal), ref=np.max)
        elif self.feature == "rms":
            return librosa.feature.rms(signal)
        elif self.feature == "zero_crossing_rate":
            return librosa.feature.zero_crossing_rate(signal)
        else:
            raise ValueError(f"Unsupported feature: {self.feature}")

    def write_gz_json(self, json_obj, filename):
        json_str = json.dumps(json_obj) + "\n"
        json_bytes = json_str.encode('utf-8')

        with gzip.GzipFile(filename, 'w') as fout:
            fout.write(json_bytes)

# Example usage
feature = 'mfcc'  # Specify the feature to extract
processor = AudioProcessor(feature)
processor.run()

