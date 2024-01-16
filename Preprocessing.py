import os
import librosa
import librosa.display
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import math
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs:")
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
import h5py
import shutil
import re
import soundfile as sf


class PathManager:
    def __init__(self):
        self.source = self.choose_source()
        self.min_size = 100000000
        self.max_size = 200000000
        self.valid_extension = '.wav'
        self.threshold = 0.01
        self.above_threshold_duration = 300
        self.required_num_files = 9
        self.feature_to_extract = 'mfcc'
        self.treatment_mapping = {
            "Gb12": "LD",
            "Gb24": "LL",
            "ALAN5": "5lux",
            "ALAN2": "2lux"
        }

        self.paths = self._fetch_paths()
        self.use_subfolders = True

    def choose_source(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Popup to decide the type of source to select
        choice = messagebox.askyesno("Source Type", "Do you want to select a directory? (No for selecting a text file)")

        if choice:  # User chose 'Yes' to select a directory
            directory = filedialog.askdirectory(title="Choose a directory")
            if directory:
                return directory
        else:  # User chose 'No', so they want to select a text file
            file_path = filedialog.askopenfilename(title="Choose a text file", filetypes=[("Text Files", "*.txt")])
            if file_path:
                return file_path

        return None

    def _fetch_paths(self):
        if os.path.isdir(self.source):
            output_file = os.path.join(os.path.dirname(self.source), "valid_folders.txt")
            with open(output_file, 'w') as file:
                self.find_valid_folders(file)
            print(f"Valid paths have been saved to {output_file}. Please use this file for further processing.")
        elif os.path.isfile(self.source) and self.source.endswith('.txt'):
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
        y, sr = librosa.load(file_path, sr=44100)
        return self._duration_above_amplitude_simple(y, sr) >= self.above_threshold_duration

    def find_valid_folders(self, file):
        for root, _, files in os.walk(self.source):
            print('you are currently in:', root)

            wav_files = [f for f in files if f.lower().endswith('.wav')]

            if len(wav_files) < self.required_num_files:
                continue

            valid_files_paths = []
            for file_name in wav_files:
                full_path = os.path.join(root, file_name)
                if self._check_file_size(full_path) and self._is_valid_audio(full_path):
                    valid_files_paths.append(full_path)
                    if len(valid_files_paths) == self.required_num_files:
                        break

            if len(valid_files_paths) >= self.required_num_files:
                print(f"Number of valid files in {root}: {len(valid_files_paths)}")
                for valid_path in valid_files_paths:
                    file.write(valid_path + '\n')

    def _read_paths_from_file(self, filepath):
        with open(filepath, 'r') as file:
            paths = [line.strip() for line in file if
                     os.path.isfile(line.strip()) and line.strip().lower().endswith('.wav')]
        return paths

    def get_treatment(self, path):
        # Iterate through all possible treatment codes
        for code, treatment in self.treatment_mapping.items():
            # If code is found anywhere in the path, return the corresponding treatment
            if code in path:
                return treatment
        # If no code is found, return None
        return None

    def create_feature_and_treatment_folders(self):

        feature_name_folder = os.path.join(os.path.dirname(self.source),
                                           self.feature_to_extract)
        os.makedirs(feature_name_folder, exist_ok=True)

        for treatment in self.treatment_mapping.values():
            os.makedirs(os.path.join(feature_name_folder, treatment), exist_ok=True)

        return feature_name_folder

    def distribute_paths_to_processes(self):
        feature_folder = self.create_feature_and_treatment_folders()

        processes = []
        for key, treatment in self.treatment_mapping.items():
            treatment_folder = os.path.join(feature_folder, treatment)
            paths_for_treatment = [path for path in self.paths if key in os.path.dirname(path)]

            # Creating an instance of AudioProcessor for each process
            audio_processor_instance = AudioProcessor(feature=self.feature_to_extract)

            # Creating process for each treatment
            process = Process(target=self.process_function,
                              args=(treatment_folder, paths_for_treatment, audio_processor_instance))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    def process_function(self, treatment_folder, paths, audio_processor_instance):
        for path in paths:
            # Decide the destination folder
            if self.use_subfolders:
                subfolder_name = os.path.basename(os.path.dirname(path))
                destination_folder = os.path.join(treatment_folder, subfolder_name)
                os.makedirs(destination_folder, exist_ok=True)
            else:
                destination_folder = treatment_folder

            treatment = os.path.basename(treatment_folder)
            print(os.path.basename(path), 'file is treated in treatment ', treatment)
            audio_processor_instance.process_file(path, treatment, destination_folder)


    def get_paths(self):
        return self.paths

    def organize_valid_files(self, destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

        if os.path.isfile(self.source) and self.source.endswith('.txt'):
            paths = self._read_paths_from_file(self.source)
        else:
            raise ValueError("The provided source must be a valid text file.")

        for path in paths:
            subject = self.extract_subject_info(path)
            treatment = self.get_treatment(path)
            if treatment:
                target_folder = os.path.join(destination_folder, treatment, subject)
                os.makedirs(target_folder, exist_ok=True)
                # shutil.copy2(path, target_folder)
                dest_path = os.path.join(target_folder, os.path.basename(path))
                # Perform pre-processing and save the result
                self.pre_process_audio(path, dest_path)
            else:
                print(f"Skipping {path}: Unable to determine treatment.")

    def extract_subject_info(self, path):
        # Define a regex pattern to extract 8 characters before '/raven.' or '\raven.'
        pattern = re.compile(r'(.{8})[\\/](raven\.)')

        # Search for the pattern in the path
        match = pattern.search(path)

        if match:
            # Extract and return the matched subject info
            subject_info = match.group(1)
            return subject_info
        else:
            # Handle case where the pattern is not found
            print(f"Warning: Expected pattern not found in {path}.")
            return None

    def remove_mean(self, audio_signal):
        return audio_signal - np.mean(audio_signal)

    def automatic_gain_control(self, audio_signal):
        return audio_signal / np.max(np.abs(audio_signal))

    def pre_process_audio(self, source_path, dest_path):
        # Load the audio file
        audio_signal, sr = librosa.load(source_path, sr=None)

        # Pre-processing
        audio_signal = self.remove_mean(audio_signal)
        audio_signal = self.automatic_gain_control(audio_signal)

        # Save the processed audio
        sf.write(dest_path, audio_signal, sr)


class AudioProcessor:
    N_MFCC = 128
    FRAME_SIZE = 2048
    HOP_LENGTH = int(FRAME_SIZE*0.25)
    NUM_SEGMENTS = 90
    SAMPLE_RATE = 44100

    def __init__(self, feature: str):
        self.feature = feature

    def process_file(self, file_path, treatment, destination):
        dict_data = self.get_directory_info(file_path, treatment)

        # Modify destination to replace .wav with .h5 in filename
        h5_path = os.path.join(destination, os.path.basename(file_path).replace(".wav", ".h5"))
        if os.path.isfile(h5_path):
            print(f"Skipping file {file_path} as it already exists.")
            return

        signal, sr = librosa.load(file_path, sr=self.SAMPLE_RATE)
        num_samples_per_segment, expected_shape = self.get_expected_shape(signal)

        for s in range(self.NUM_SEGMENTS):
            segment_signal = self.get_segment_signal(file_path, s, signal, num_samples_per_segment)

            if segment_signal is not None:
                feature_vectors = self.extract_feature(segment_signal)

                if self.check_feature_vectors(feature_vectors, expected_shape, file_path, s):
                    self.update_dict_data(dict_data, feature_vectors)

        self.write_h5(dict_data, h5_path)

    def update_dict_data(self, dict_data, feature_vectors):
        dict_data[self.feature].append(feature_vectors.tolist())

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
        samples_above_threshold = np.sum(np.abs(segment_signal) > 0.001)

        if samples_above_threshold < (0.3 * len(segment_signal)):  # You can adjust this value based on your requirement
            print(
                f"More than half of the segment {s} of file {filename} is below the threshold. Skipping this segment.")
            return None
        return segment_signal

    def get_directory_info(self, file_path, treatment):

        if not os.path.isfile(file_path):
            print(f"Error: {file_path} is not a valid file.")
            return None, None, None, None

        filename = os.path.basename(file_path)
        if not filename.lower().endswith('.wav'):
            print(f"Error: {filename} is not a .wav file.")
            return None, None, None, None

        dict_data = {
            "path": file_path,
            self.feature: [],
            "labels": [treatment] * self.NUM_SEGMENTS,
            "segment_number": list(range(self.NUM_SEGMENTS))
        }
        return dict_data

    @staticmethod
    def write_h5(data, filename):
        with h5py.File(filename, 'w') as hf:
            for key, value in data.items():
                if isinstance(value, str):  # Check if the value is a string
                    encoded_value = value.encode('utf-8')  # Convert string to byte string
                    hf.create_dataset(key, data=encoded_value)
                elif isinstance(value, list) and all(
                        isinstance(item, str) for item in value):  # Check if the value is a list of strings
                    str_dtype = h5py.string_dtype(encoding='utf-8')
                    hf.create_dataset(key, data=value, dtype=str_dtype)
                else:
                    hf.create_dataset(key, data=np.array(value))

    def get_expected_shape(self, signal):

        duration = librosa.get_duration(y=signal, sr=self.SAMPLE_RATE)
        samples_per_track = self.SAMPLE_RATE * duration
        num_samples_per_segment = samples_per_track / self.NUM_SEGMENTS

        if self.feature in ["mfcc", "spectrogram", "mel_spectrogram"]:
            expected_shape = (self.N_MFCC, math.ceil(num_samples_per_segment / self.HOP_LENGTH))
        elif self.feature in ["ae", "rms", "zcr", "ber", 'sc', 'bw']:
            expected_shape = (1, math.ceil(num_samples_per_segment / self.HOP_LENGTH))
        elif self.feature in ["chroma"]:
            expected_shape = (12, math.ceil(num_samples_per_segment / self.HOP_LENGTH))
        elif self.feature in ['mfccs_and_derivatives']:
            expected_shape = (self.N_MFCC*3, math.ceil(num_samples_per_segment / self.HOP_LENGTH))
        else:
            raise ValueError(f"Unsupported feature: {self.feature}")

        return num_samples_per_segment, expected_shape

    def extract_feature(self, signal):

        if self.feature == "mfcc":
            return librosa.feature.mfcc(y=signal, n_fft=self.FRAME_SIZE, n_mfcc=self.N_MFCC, hop_length=self.HOP_LENGTH,
                                        sr=self.SAMPLE_RATE)
        elif self.feature == "mfccs_and_derivatives":
            mfccs = librosa.feature.mfcc(y=signal, n_fft=self.FRAME_SIZE, n_mfcc=self.N_MFCC,
                                         hop_length=self.HOP_LENGTH, sr=self.SAMPLE_RATE)

            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            combined = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
            return combined

        elif self.feature == "chroma":
            return librosa.feature.chroma_stft(y=signal, sr=self.SAMPLE_RATE, n_fft=self.FRAME_SIZE,
                                               hop_length=self.HOP_LENGTH)


        elif self.feature == "spectrogram":

            S_complex = librosa.stft(signal, hop_length=self.HOP_LENGTH, n_fft=self.FRAME_SIZE)
            spectrogram = np.abs(S_complex) + 1e-5  # Adding a small constant to avoid log(0)
            log_spectrogram = librosa.amplitude_to_db(spectrogram)
            normalized_log_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / (
                        np.max(log_spectrogram) - np.min(log_spectrogram))  # Normalization
            print(normalized_log_spectrogram.shape)
            return normalized_log_spectrogram
        elif self.feature == "mel_spectrogram":
            S = librosa.feature.melspectrogram(y=signal, sr=self.SAMPLE_RATE)
            S_dB = librosa.power_to_db(S, ref=np.max)

            return S_dB
        elif self.feature == "ae":
            amplitude_envelope = []
            for i in range(0, len(signal), self.HOP_LENGTH):
                current_ae = max(signal[i:i+self.FRAME_SIZE])
                amplitude_envelope.append(current_ae)
            amplitude_envelope = np.array(amplitude_envelope).reshape(1, -1)
            return amplitude_envelope
        elif self.feature == "rms":
            return librosa.feature.rms(y=signal, frame_length=self.FRAME_SIZE, hop_length=self.HOP_LENGTH)
        elif self.feature == "zcr":
            return librosa.feature.zero_crossing_rate(signal, frame_length=self.FRAME_SIZE, hop_length=self.HOP_LENGTH)
        elif self.feature == "ber":
            spectogram = librosa.stft(signal, n_fft=self.FRAME_SIZE, hop_length=self.HOP_LENGTH)
            frequency_range = self.SAMPLE_RATE/2
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
            return librosa.feature.spectral_centroid(y=signal, sr=self.SAMPLE_RATE, n_fft=self.FRAME_SIZE, hop_length=self.HOP_LENGTH)
        elif self.feature == 'bw':
            return librosa.feature.spectral_bandwidth(y=signal, sr=self.SAMPLE_RATE, n_fft=self.FRAME_SIZE, hop_length=self.HOP_LENGTH)
        else:
            raise ValueError(f"Unsupported feature: {self.feature}")



if __name__ == '__main__':
    path_manager = PathManager()
    destination_folder = filedialog.askdirectory()
    path_manager.organize_valid_files(destination_folder)
    # path_manager.distribute_paths_to_processes()