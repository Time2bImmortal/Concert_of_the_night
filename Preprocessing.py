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
import multiprocessing
import glob
from typing import Tuple
import supporting_functions
"""
Here, we preprocess audio files that are already organized in a specific folder structure: 
Folder > Treatments > Subfolders > Audio files.

The main objective is to create condensed gzip files that contain the extracted sound features, along with the
corresponding file name, subfolder, treatment, and format information."""


def process_treatment(audio_processor, treatment):
    treatment_dir = os.path.join(audio_processor.features_dir, treatment)
    audio_files = audio_processor.audio_files_dict[treatment]
    for file in audio_files:
        print('file:', file)
        audio_processor.process_file(file)


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


class AudioProcessor:
    def __init__(self, feature: str, src_directory,
                 n_mfcc=13, frame_size=2048, hop_length=1024, num_segments=30):

        self.feature = feature
        self.src_directory = src_directory
        self.file_extension = ".gz"
        self.n_mfcc = n_mfcc
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.num_segments = num_segments
        self.treatments = os.listdir(src_directory)
        self.treatments_dir = []
        self.features_dir = None

    def create_feature_directory(self):

        feature_directory = os.path.join(os.path.dirname(self.src_directory), self.feature)
        os.makedirs(feature_directory, exist_ok=True)
        return feature_directory

    def create_treatment_directories(self):

        for treatment in self.treatments:
            treatment_dir = os.path.join(self.features_dir, treatment)
            self.treatments_dir.append(treatment_dir)
            os.makedirs(treatment_dir, exist_ok=True)

    def run(self):

        self.features_dir = self.create_feature_directory()
        self.create_treatment_directories()

        # Calculate the minimum number of .wav files across all treatment directories
        min_file_count = min(
            len(glob.glob(os.path.join(self.src_directory, treatment, '**/*.wav'), recursive=True))
            for treatment in self.treatments
        )

        processes = []
        for i, treatment in enumerate(self.treatments):
            treatment_dir_in_src = os.path.join(self.src_directory, treatment)
            treatment_dir_in_features = os.path.join(self.features_dir, treatment)
            process = multiprocessing.Process(
                target=self.process_treatment,
                args=(treatment_dir_in_src, treatment_dir_in_features, min_file_count,))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    def process_treatment(self, treatment_dir_in_src: str, treatment_dir_in_features: str, file_count: int) -> None:

        wav_files = glob.glob(os.path.join(treatment_dir_in_src, '**/*.wav'), recursive=True)
        wav_files = wav_files[:file_count]  # Only take the first `file_count` files
        for counter, file_path in enumerate(wav_files, start=1):
            self.process_file(file_path, counter, treatment_dir_in_features)

    def process_file(self, file_path, counter, treatment_dir_features):

        filename, subfolder, treatment, dict_data = self.get_directory_info(file_path)
        print(filename, "is being processed in", treatment)
        gz_file_return = self.create_gz_file(filename, counter, subfolder, treatment_dir_features)
        if gz_file_return is None:
            print(f"Skipping file {file_path} as it already exists.")
            return
        gz_json_file, gz_json_path = gz_file_return
        signal, sr = librosa.load(file_path, sr=44100)
        num_samples_per_segment, expected_shape = self.get_expected_shape(signal, sr)

        for s in range(self.num_segments):
            segment_signal = self.get_segment_signal(filename, s, signal, num_samples_per_segment)

            if segment_signal is not None:
                feature_vectors = self.extract_feature(segment_signal, sr)

                if self.check_feature_vectors(feature_vectors, expected_shape, filename, s):
                    self.update_dict_data(dict_data, feature_vectors, treatment, s)

        self.write_gz_json(dict_data, gz_json_path)
        self.write_metadata(sr, signal, feature_vectors)

    def write_metadata(self, sr, signal, feature_vectors):

        metadata_file_path = os.path.join(self.features_dir, f'{self.feature}_metadata.txt')
        if not os.path.exists(metadata_file_path):
            details = self.get_details(sr, signal, feature_vectors)
            with open(metadata_file_path, 'w') as f:
                f.write(f"Extraction details for processed feature: {self.feature}:\n")
                for key, value in details.items():
                    f.write(f"{key}: {value}\n")

    def get_details(self, sr, signal, feature_vectors):

        treatment_indices = {treatment: i for i, treatment in enumerate(self.treatments)}

        return {
            "Sampling rate": sr,
            "Number of segments": self.num_segments,
            "Duration": librosa.get_duration(y=signal, sr=sr),
            "Hop length": self.hop_length,
            "Frame size": self.frame_size,
            "Shape of the feature extracted": feature_vectors.shape,
            "Treatments and indices": treatment_indices
        }

    def update_dict_data(self, dict_data, feature_vectors, treatment, s):

        if treatment in self.treatments:
            dict_data[self.feature].append(feature_vectors.tolist())
            dict_data["labels"].append(self.treatments.index(treatment))
            dict_data["segment_number"].append(s)
        else:
            print(f"Treatment {treatment} not found in self.treatments")

    def check_feature_vectors(self, feature_vectors, expected_shape, filename, s):

        if len(feature_vectors) > 0:
            if feature_vectors.shape != expected_shape:
                with open('problem_files.txt', 'a') as f:
                    f.write(
                        f'File {filename} section {s} produced feature vector with shape {feature_vectors.shape}.\n')
                return False
            return True
        return False

    def get_segment_signal(self, filename, s, signal, num_samples_per_segment):

        start_sample = int(num_samples_per_segment * s)
        end_sample = int(start_sample + num_samples_per_segment)
        segment_signal = signal[start_sample:end_sample]

        if len(segment_signal) == 0:
            print(f"Empty segment signal at segment {s} of file {filename}. Skipping this segment.")
            return None
        return segment_signal

    def create_gz_file(self, filename, counter, subfolder, treatment_dir_features):

        gz_json_file = filename.replace(filename, f"{str(counter).zfill(5)}_{subfolder}_{self.feature}.gz")
        gz_json_path = os.path.join(treatment_dir_features, gz_json_file)
        if os.path.isfile(gz_json_path):
            print(f"'{gz_json_file}' file already exists in {treatment_dir_features}. Skipping this file.")
            return None
        return gz_json_file, gz_json_path

    def get_directory_info(self, file_path):

        if not os.path.isfile(file_path):
            print(f"Error: {file_path} is not a valid file.")
            return None, None, None, None

        filename = os.path.basename(file_path)
        if not filename.lower().endswith('.wav'):
            print(f"Error: {filename} is not a .wav file.")
            return None, None, None, None

        relative_path = os.path.relpath(file_path, self.src_directory)
        parts = relative_path.split(os.path.sep)
        treatment = parts[0] if len(parts) > 1 else None
        subfolder = parts[1] if len(parts) > 2 else None
        dict_data = {
            "path": file_path,
            "subfolder_name": subfolder,
            "filename": filename,
            self.feature: [],
            "labels": [],
            "segment_number": []
        }
        return filename, subfolder, treatment, dict_data

    def get_expected_shape(self, signal, sr):

        duration = librosa.get_duration(y=signal, sr=sr)
        samples_per_track = sr * duration
        num_samples_per_segment = samples_per_track / self.num_segments

        if self.feature in ["mfcc", "spectrogram", "mel_spectrogram"]:
            expected_shape = (self.n_mfcc, math.ceil(num_samples_per_segment / self.hop_length))
        elif self.feature in ["ae", "rms", "zcr", "ber", 'sc', 'bw']:
            expected_shape = (1, math.ceil(num_samples_per_segment / self.hop_length))
        else:
            raise ValueError(f"Unsupported feature: {self.feature}")

        return num_samples_per_segment, expected_shape

    def extract_feature(self, signal, sr):

        if self.feature == "mfcc":
            return librosa.feature.mfcc(y=signal, n_fft=self.frame_size, n_mfcc=self.n_mfcc, hop_length=self.hop_length,
                                        sr=sr)
        elif self.feature == "spectrogram":
            S_complex = librosa.stft(signal, hop_length=self.hop_length, n_fft=self.frame_size)
            spectrogram = np.abs(S_complex)
            log_spectrogram = librosa.amplitude_to_db(spectrogram)
            return log_spectrogram
        elif self.feature == "mel_spectrogram":
            S = librosa.feature.melspectrogram(y=signal, sr=sr)
            return S
        elif self.feature == "ae":
            amplitude_envelope = []
            for i in range(0, len(signal), self.hop_length):
                current_ae = max(signal[i:i+self.frame_size])
                amplitude_envelope.append(current_ae)
            amplitude_envelope = np.array(amplitude_envelope).reshape(1, -1)
            return amplitude_envelope
        elif self.feature == "rms":
            return librosa.feature.rms(y=signal, frame_length=self.frame_size, hop_length=self.hop_length)
        elif self.feature == "zcr":
            return librosa.feature.zero_crossing_rate(signal, frame_length=self.frame_size, hop_length=self.hop_length)
        elif self.feature == "ber":
            spectogram = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)
            frequency_range = sr/2
            frequency_delta_bin = frequency_range/spectogram.shape[0]
            split_frequency = int(np.floor(2000/frequency_delta_bin))
            power_spec = np.abs(spectogram) ** 2
            power_spec = power_spec.T
            band_energy_ratio = []
            for frequencies_in_frame in power_spec:
                sum_power_spec_low = np.sum(frequencies_in_frame[:split_frequency])
                sum_power_spec_high = np.sum(frequencies_in_frame[split_frequency:])
                ber_current_frame = sum_power_spec_low/sum_power_spec_high
                band_energy_ratio.append(ber_current_frame)
            return np.array(band_energy_ratio).reshape(1, -1)
        elif self.feature == 'sc':
            return librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=self.frame_size, hop_length=self.hop_length)
        elif self.feature == 'bw':
            return librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=self.frame_size, hop_length=self.hop_length)
        else:
            raise ValueError(f"Unsupported feature: {self.feature}")

    def write_gz_json(self, json_obj, filename):

        json_str = json.dumps(json_obj) + "\n"
        json_bytes = json_str.encode('utf-8')

        with gzip.GzipFile(filename, 'w') as fout:
            fout.write(json_bytes)

