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
import multiprocessing
from typing import Tuple
FEATURE_ABBREVIATIONS = {
    "amplitude_envelope": "ae",
    "root_mean_square": "rms",
    "zero_crossing_rate": "zcr",
    "spectral_bandwidth": "bw",
    "spectral_centroid": "sc",
    "band_energy_ration": "ber"
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

def process_treatment(audio_processor, treatment):
    treatment_dir = os.path.join(audio_processor.features_dir, treatment)
    audio_files = audio_processor.audio_files_dict[treatment]
    for file in audio_files:
        print('file:', file)
        audio_processor.process_file(file)


class AudioProcessor:
    def __init__(self, feature: str, src_directory,
                 n_mfcc=13, frame_size=2048, hop_length=1024, num_segments=30):
        self.feature = feature  # Feature to extract from audio files
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
        processes = []
        # Loop through the treatment directories in the original source directory
        for i, treatment in enumerate(self.treatments):
            treatment_dir_in_src = os.path.join(self.src_directory, treatment)  # Path in the original source directory
            treatment_dir_in_features = os.path.join(self.features_dir, treatment)  # Path in the features_dir
            process = multiprocessing.Process(target=self.process_treatment,
                                              args=(treatment_dir_in_src, treatment_dir_in_features,))
            process.start()
            processes.append(process)

        # Ensuring all processes have finished execution
        for process in processes:
            process.join()

    def process_treatment(self, treatment_dir_in_src: str, treatment_dir_in_features: str) -> None:
        # Iterate through the treatment directory and any subdirectories
        for subdir, dirs, files in os.walk(treatment_dir_in_src):
            for counter, file in enumerate(files, start=1):
                if file.endswith('.wav'):
                    file_path = os.path.join(subdir, file)
                    # Save results in treatment_dir_in_features
                    self.process_file(file_path, counter, treatment_dir_in_features)

    def process_file(self, file_path, counter, treatment_dir_features):
        # existing logic
        filename, subfolder, treatment, dict_data = self.get_directory_info(file_path)
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
        return {
            "Sampling rate": sr,
            "Number of segments": self.num_segments,
            "Duration": librosa.get_duration(y=signal, sr=sr),
            "Hop length": self.hop_length,
            "Frame size": self.frame_size,
            "Shape of the feature extracted": feature_vectors.shape,
        }

    def update_dict_data(self, dict_data, feature_vectors, treatment, s):
        if treatment in self.treatments:
            dict_data[self.feature].append(feature_vectors.tolist())
            dict_data["labels"].append(self.treatments.index(treatment))
            dict_data["segment_number"].append(s)
        else:
            print(f"Treatment {treatment} not found in self.treatments")

    def check_feature_vectors(self, feature_vectors, expected_shape, filename, s):
        if len(feature_vectors) > 0:  # Check if feature_vectors is not empty
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

        # Check if the file is a .wav file
        filename = os.path.basename(file_path)
        if not filename.lower().endswith('.wav'):
            print(f"Error: {filename} is not a .wav file.")
            return None, None, None, None

        # Extract subfolder and treatment from file_path
        relative_path = os.path.relpath(file_path, self.src_directory)
        parts = relative_path.split(os.path.sep)

        # Assuming the structure is /treatment/subfolder/filename.wav
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
        if self.feature == "mfcc": # Time-frequency feature
            return librosa.feature.mfcc(y=signal, n_fft=self.frame_size, n_mfcc=self.n_mfcc, hop_length=self.hop_length,
                                        sr=sr)
        # mfcc_delta = librosa.feature.delta(mfcc)
        # Compute second derivative (delta-delta) of MFCC
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
            amplitude_envelope = np.array(amplitude_envelope).reshape(1, -1)
            return amplitude_envelope
        elif self.feature == "rms":
            return librosa.feature.rms(y=signal, frame_length=self.frame_size, hop_length=self.hop_length)

        elif self.feature == "zcr":
            return librosa.feature.zero_crossing_rate(signal, frame_length=self.frame_size, hop_length=self.hop_length)

        # Below are frequency feature
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

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    src_directory = filedialog.askdirectory(title="Select Source Directory")
    root.destroy()

    for value in FEATURE_ABBREVIATIONS.values():
        processor = AudioProcessor(value, src_directory)
        processor.run()