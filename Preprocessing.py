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

def open_and_show_gz_file():
    tk.Tk().withdraw()  # Hide the Tkinter root window

    file = filedialog.askopenfile(filetypes=[('GZ files', '*.gz')])
    if file is None:
        print("No file selected.")
        return

    try:
        with gzip.open(file.name, 'rt') as gz_file:
            content = gz_file.read()
            print(content)
    except FileNotFoundError:
        print("File not found.")
    except gzip.BadGzipFile:
        print("Invalid .gz file.")
    except Exception as e:
        print("An error occurred:", str(e))


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


# def save_and_compare_audio(filename):
#     # Load the audio file
#     signal, sr = librosa.load(filename)
#
#     # Save the audio data to a json file
#     json_filename = filename.replace('.wav', '.json')
#     with open(json_filename, 'w') as json_file:
#         json.dump(signal.tolist(), json_file)
#
#     # Save the audio data to a gzipped json file
#     gz_filename = filename.replace('.wav', '.gz')
#     write_gz_json(signal.tolist(), gz_filename)
#
#     # Load the json data
#     with open(json_filename, 'r') as json_file:
#         json_data = np.array(json.load(json_file))
#
#     # Load the gzipped json data
#     with gzip.GzipFile(gz_filename, 'r') as gz_file:
#         gz_data = np.array(json.loads(gz_file.read().decode('utf-8')))
#
#     # Compare the two data arrays
#     if np.array_equal(json_data, gz_data):
#         print("The two files contain identical data.")
#     else:
#         print("The two files do not contain identical data.")


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
                 n_mfcc=13, frame_size=2048, hop_length=1024, num_segments=30):
        self.feature = feature  # Feature to extract from audio files
        self.file_extension = ".gz"
        self.n_mfcc = n_mfcc
        self.frame_size = frame_size
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
        self.process_directory(src_directory)
        # self.process_file(self.choose_file())

    def process_directory(self, src_directory):

        # Create directories for each feature
        self.features_dir = self.create_feature_directory(src_directory)
        self.create_treatment_directories(src_directory)

        # Iterate through files in source directory
        for subdir, dirs, files in os.walk(src_directory):
            for file in files:
                # Check if file is an audio file
                if file.endswith('.wav'):
                    # Process each audio file
                    self.process_file(os.path.join(subdir, file))

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

        signal, sr = librosa.load(file_path, sr=44100)
        duration = librosa.get_duration(y=signal, sr=sr)
        samples_per_track = sr * duration
        num_samples_per_segment = samples_per_track / self.num_segments
        expected_vectors_mfcc = math.ceil(num_samples_per_segment / self.hop_length)
        expected_shape = (math.ceil(num_samples_per_segment / self.hop_length), )
        for s in range(self.num_segments):
            start_sample = int(num_samples_per_segment * s)
            end_sample = int(start_sample + num_samples_per_segment)
            segment_signal = signal[start_sample:end_sample]

            if len(segment_signal) == 0:
                print(f"Empty segment signal at segment {s} of file {file_path}. Skipping this segment.")
                continue

            feature_vectors = self.extract_feature(segment_signal, sr)
            print(f"The file {filename}, section: {s} is being processed...")

            if len(feature_vectors) > 0:  # Check if feature_vectors is not empty
                if feature_vectors.shape != expected_shape:
                    with open('problem_files.txt', 'a') as f:
                        f.write(
                            f'File {filename} section {s} produced feature vector with shape {feature_vectors.shape}.\n')
                    continue
                dict_data[self.feature].append(feature_vectors.tolist())
                dict_data["labels"].append(self.treatments.index(treatment))
                dict_data["segment_number"].append(s)

        self.write_gz_json(dict_data, gz_json_path)

        print(f"The file {filename} has been processed.")

    def extract_feature(self, signal, sr):
        if self.feature == "mfcc": # Time-frequency feature
            return librosa.feature.mfcc(y=signal, n_fft=self.frame_size, n_mfcc=self.n_mfcc, hop_length=self.hop_length,
                                        sr=sr)
        # mfcc_delta = librosa.feature.delta(mfcc)
        #
        # # Compute second derivative (delta-delta) of MFCC
        # mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        elif self.feature == "spectrogram":
            # Compute the spectrogram magnitude and phase
            S_complex = librosa.stft(signal, hop_length=self.hop_length, n_fft=self.frame_size)
            spectogram = np.abs(S_complex)
            log_spectogram = librosa.amplitude_to_db(spectogram)
            return log_spectogram
        elif self.feature == "mel_spectrogram":
            # Compute a mel-scaled spectrogram.
            S = librosa.feature.melspectrogram(y=signal, sr=sr)
            return S
        # Below are time features
        elif self.feature == "ae":
            amplitude_envelope = []
            for i in range(0, len(signal), self.hop_length):
                current_ae = max(signal[i:i+self.frame_size])
                amplitude_envelope.append(current_ae)
            amplitude_envelope = np.array(amplitude_envelope)
            return amplitude_envelope
        elif self.feature == "rms":
            return librosa.feature.rms(signal)
        elif self.feature == "zcr":
            return librosa.feature.zero_crossing_rate(signal)
        # Below are frequency feature
        elif self.feature == "ber":
            spectogram = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)
            frequency_range = sr/2
            frequency_delta_bin = frequency_range/spectogram.shape[0]
            split_frequency = int(np.floor(2000/frequency_delta_bin))
            power_spec = np.abs(spectogram) ** 2
            power_spec = power_spec.T
            band_energy_ration = []
            for frequencies_in_frame in power_spec:
                sum_power_spec_low = np.sum(frequencies_in_frame[:split_frequency])
                sum_power_spec_high = np.sum(frequencies_in_frame[split_frequency:])
                ber_current_frame = sum_power_spec_low/sum_power_spec_high
                band_energy_ration.append(ber_current_frame)
            return np.array(band_energy_ration)
        elif self.feature == 'sc':
            return librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=self.frame_size, hop_length=self.hop_length)[0]
        elif self.feature == 'bw':
            return librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=self.frame_size, hop_length=self.hop_length)[0]
        else:
            raise ValueError(f"Unsupported feature: {self.feature}")

    def write_gz_json(self, json_obj, filename):
        json_str = json.dumps(json_obj) + "\n"
        json_bytes = json_str.encode('utf-8')

        with gzip.GzipFile(filename, 'w') as fout:
            fout.write(json_bytes)

# Example usage
# feature = 'ae'  # Specify the feature to extract
# processor = AudioProcessor(feature)
# processor.run()
open_and_show_gz_file()