import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog
import math
import collections
import shutil
from typing import List
import multiprocessing
import glob
from typing import Tuple
import supporting_functions
import h5py

class PathManager:
    def __init__(self):
        self.source = self.choose_source()
        self.min_size = 100000000
        self.max_size = 200000000
        self.valid_extension = '.wav'
        self.threshold = 0.01
        self.above_threshold_duration = 300
        self.required_num_files = 20

        self.treatment_mapping = {
            "Gb12": "LD",
            "Gb24": "LL",
            "ALAN5": "5lux",
            "ALAN2": "2lux"
        }

        self.paths = self._fetch_paths()

    def choose_source(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Ask user to choose either a file or a directory
        file_or_directory = filedialog.askopenfilename(title="Select a file or directory",
                                                       defaultextension=("Text Files", "*.txt"),
                                                       filetypes=(("Text Files", "*.txt"),("All Files", "*.*")))
        # If user closed or canceled, then ask for directory
        if not file_or_directory:
            file_or_directory = filedialog.askdirectory(title="Choose a directory")

        return file_or_directory

    def _fetch_paths(self):
        if os.path.isdir(self.source):
            self.find_valid_folders()
            output_file = os.path.join(os.path.dirname(self.source), "valid_folders.txt")
            return self._read_paths_from_file(output_file)
        elif os.path.isfile(self.source):
            return self._read_paths_from_file(self.source)
        else:
            raise ValueError("The provided source is neither a directory nor a valid file.")

    def _check_file_size(self, file_path):
        file_size = os.path.getsize(file_path)
        return self.min_size <= file_size <= self.max_size

    def _duration_above_amplitude_simple(self, y, sr):
        samples_above_threshold = np.sum(np.abs(y) > self.threshold)
        return samples_above_threshold / sr

    def _is_valid_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        return self._duration_above_amplitude_simple(y, sr) >= self.above_threshold_duration

    def find_valid_folders(self):
        output_file = os.path.join(os.path.dirname(self.source), "valid_folders.txt")

        with open(output_file, 'w') as file:
            for root, _, files in os.walk(self.source):
                valid_files = [os.path.join(root, file_name) for file_name in files if
                               file_name.lower().endswith(self.valid_extension) and
                               self._check_file_size(os.path.join(root, file_name)) and
                               self._is_valid_audio(os.path.join(root, file_name))]

                if len(valid_files) >= self.required_num_files:
                    file.write(root + '\n')
                    for valid_file in valid_files:
                        file.write(valid_file + '\n')
                    file.write('\n')  # Add an extra newline for clarity

        print(f"Paths written to {output_file}.")

    def _read_paths_from_file(self, filepath):
        with open(filepath, 'r') as file:
            paths = []
            for line in file:
                line = line.strip()
                if os.path.isfile(line):
                    paths.append(line)
                elif os.path.isdir(line):
                    paths.extend([os.path.join(line, file) for file in os.listdir(line) if file.endswith(self.valid_extension)])
            return paths

    def get_treatment(self, path):
        base_name = os.path.basename(path).split('.')[0]  # Remove extension to get the name
        return self.treatment_mapping.get(base_name, None)

    def create_nested_folder(self, path):
        treatment = self.get_treatment(path)
        if not treatment:
            print(f"Warning: No treatment found for {path}.")
            return

        nested_folder_path = os.path.join(os.path.dirname(path), treatment)
        os.makedirs(nested_folder_path, exist_ok=True)  # Safely create nested directory

        return nested_folder_path

    def get_paths(self):
        return self.paths


class AudioProcessor:
    def __init__(self, feature: str):

        self.feature = feature
        self.n_mfcc = 13
        self.frame_size = 2048
        self.hop_length = 1024
        self.num_segments = 30
        self.sample_rate = 44100

    def process_file(self, file_path, treatment_dir_features):

        filename, treatment, dict_data = self.get_directory_info(file_path)
        print(filename, "is being processed in", treatment)
        h5_file_return = self.create_h5_file(filename, treatment_dir_features)
        if h5_file_return is None:
            print(f"Skipping file {file_path} as it already exists.")
            return
        h5_file, h5_path = h5_file_return
        signal, sr = librosa.load(file_path, sr=self.sample_rate)
        num_samples_per_segment, expected_shape = self.get_expected_shape(signal, sr)

        for s in range(self.num_segments):
            segment_signal = self.get_segment_signal(filename, s, signal, num_samples_per_segment)

            if segment_signal is not None:
                feature_vectors = self.extract_feature(segment_signal, sr)

                if self.check_feature_vectors(feature_vectors, expected_shape, filename, s):
                    self.update_dict_data(dict_data, feature_vectors, treatment, s)

        self.write_h5(dict_data, h5_path)
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

    def create_h5_file(self, filename, counter, treatment_dir_features):

        h5_file = filename.replace(filename, f"{str(counter).zfill(5)}_{self.feature}.h5")
        h5_path = os.path.join(treatment_dir_features, h5_file)
        if os.path.isfile(h5_path):
            print(f"'{h5_file}' file already exists in {treatment_dir_features}. Skipping this file.")
            return None
        return h5_file, h5_path

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
        dict_data = {
            "path": file_path,
            "filename": filename,
            self.feature: [],
            "labels": [],
            "segment_number": []
        }
        return filename, treatment, dict_data

    def get_expected_shape(self, signal, sr):

        duration = librosa.get_duration(y=signal, sr=sr)
        samples_per_track = sr * duration
        num_samples_per_segment = samples_per_track / self.num_segments

        if self.feature in ["mfcc", "spectrogram", "mel_spectrogram"]:
            expected_shape = (self.n_mfcc, math.ceil(num_samples_per_segment / self.hop_length))
        elif self.feature in ["ae", "rms", "zcr", "ber", 'sc', 'bw']:
            expected_shape = (1, math.ceil(num_samples_per_segment / self.hop_length))
        elif self.feature in ['mfccs_and_derivatives']:
            expected_shape = (self.n_mfcc*3, math.ceil(num_samples_per_segment / self.hop_length))
        else:
            raise ValueError(f"Unsupported feature: {self.feature}")

        return num_samples_per_segment, expected_shape

    def extract_feature(self, signal, sr):

        if self.feature == "mfcc":
            return librosa.feature.mfcc(y=signal, n_fft=self.frame_size, n_mfcc=self.n_mfcc, hop_length=self.hop_length,
                                        sr=sr)
        elif self.feature == "mfccs_and_derivatives":
            mfccs = librosa.feature.mfcc(y=signal, n_fft=self.frame_size, n_mfcc=self.n_mfcc,
                                         hop_length=self.hop_length, sr=sr)

            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            combined = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
            return combined
        elif self.feature == "spectrogram":
            S_complex = librosa.stft(signal, hop_length=self.hop_length, n_fft=self.frame_size)
            spectrogram = np.abs(S_complex)
            log_spectrogram = librosa.amplitude_to_db(spectrogram)
            return log_spectrogram
        elif self.feature == "mel_spectrogram":
            S = librosa.feature.melspectrogram(y=signal, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)

            return S_dB
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

    def write_h5(self, data, filename):
        with h5py.File(filename, 'w') as hf:
            # Assuming data is a dictionary where keys are dataset names and values are numpy arrays
            for key, value in data.items():

                if isinstance(value, str):  # Check if the value is a string
                    encoded_value = value.encode('utf-8')  # Convert string to byte string
                    hf.create_dataset(key, data=np.array(encoded_value))
                else:
                    hf.create_dataset(key, data=np.array(value))


if __name__ == '__main__':
    path_manager = PathManager()

