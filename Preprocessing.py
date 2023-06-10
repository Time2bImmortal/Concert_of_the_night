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


# class AudioProcessor:
#
#     def __init__(self, src_directory: str, dst_directory: str, treatments: List[str], feature: str):
#         self.src_directory = src_directory
#         self.dst_directory = dst_directory
#         self.treatments = treatments
#         self.feature = feature  # Feature to extract from audio files
#         self.file_extension = ".gz"
#
#     def process_single_file(self, treatment, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=30):
#         # Instantiate a Tk instance (required to open a dialog window)
#         root = tk.Tk()
#         # Hide the main window
#         root.withdraw()
#         # Open a file selection dialog
#         file_path = filedialog.askopenfilename()
#         # Close the Tk instance
#         root.destroy()
#
#         # Check if the user selected a file
#         if file_path:
#             filename = os.path.basename(file_path)
#             subfolder_path = os.path.dirname(file_path)
#             self.process_file(filename, subfolder_path, treatment, n_mfcc, n_fft, hop_length, num_segments)
#
#             gz_json_file = filename.replace(".wav", f"_{self.feature}.gz")  # append feature to filename
#             gz_json_path = os.path.join(subfolder_path, gz_json_file)
#
#             with gzip.GzipFile(gz_json_path, 'r') as gz_file:
#                 dict_data = json.loads(gz_file.read().decode('utf-8'))
#
#             print(dict_data)
#         else:
#             print("No file selected.")
#     def run(self):
#         self.organize_files_by_treatment()
#         self.save_features()
#         self.combine_dicts()
#         self.open_and_view_json()
#
#     def organize_files_by_treatment(self):
#         assert self.src_directory != self.dst_directory, "Source and destination directories cannot be the same."
#         for treatment in self.treatments:
#             treatment_directory = os.path.join(self.dst_directory, treatment + "_extract")
#             os.makedirs(treatment_directory, exist_ok=True)
#             for dirpath, dirnames, filenames in os.walk(self.src_directory):
#                 for filename in filenames:
#                     if filename.endswith(self.file_extension) and treatment in dirpath:
#                         src_file_path = os.path.join(dirpath, filename)
#                         dst_file_path = os.path.join(treatment_directory, filename)
#                         shutil.copy2(src_file_path, dst_file_path)
#
#     def extract_feature(self, segment_signal, sr):
#         if self.feature == 'mfcc':
#             return librosa.feature.mfcc(y=segment_signal, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
#         elif self.feature == 'rms':
#             return librosa.feature.rms(y=segment_signal)
#         elif self.feature == 'ae':
#             return librosa.feature.amplitude_envelope(y=segment_signal, sr=sr)
#         else:
#             raise ValueError(f"Invalid feature: {self.feature}")
#
#     def save_features(self, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=30):
#         treatments = sorted([dir_name for dir_name in os.listdir(self.src_directory) if
#                              os.path.isdir(os.path.join(self.src_directory, dir_name))])
#
#         subfolders_by_treatment = collections.defaultdict(list)
#         for treatment in treatments:
#             treatment_path = os.path.join(self.src_directory, treatment)
#             subfolders = sorted(
#                 [name for name in os.listdir(treatment_path) if os.path.isdir(os.path.join(treatment_path, name))])
#             subfolders_by_treatment[treatment] = subfolders
#
#         max_subfolders = max(len(subfolders) for subfolders in subfolders_by_treatment.values())
#
#         for i in range(max_subfolders):
#             for treatment in treatments:
#                 subfolders = subfolders_by_treatment[treatment]
#                 if i < len(subfolders):
#                     subfolder = subfolders[i]
#                     subfolder_path = os.path.join(self.src_directory, treatment, subfolder)
#
#                     for filename in os.listdir(subfolder_path):
#                         if filename.endswith(".wav"):
#                             self.process_file(filename, subfolder_path, treatment, n_mfcc, n_fft, hop_length,
#                                               num_segments)
#
#     def process_file(self, filename, subfolder_path, treatment, n_mfcc, n_fft, hop_length, num_segments):
#         file_path = os.path.join(subfolder_path, filename)
#
#         dict_data = {
#             "light_treatment": [],
#             self.feature: [],  # use selected feature instead of 'mfcc'
#             "labels": [],
#             "subfolder_name": [],
#             "segment_number": []
#         }
#
#         gz_json_file = filename.replace(".wav", f"_{self.feature}.gz")  # append feature to filename
#         gz_json_path = os.path.join(subfolder_path, gz_json_file)
#         if os.path.isfile(gz_json_path):
#             print(f"'{gz_json_file}' file already exists in {subfolder_path}. Skipping this file.")
#             return
#
#         signal, sr = librosa.load(file_path)
#         duration = librosa.get_duration(y=signal, sr=sr)
#         samples_per_track = sr * duration
#         num_samples_per_segment = samples_per_track / num_segments
#         expected_vectors_mfcc = math.ceil(num_samples_per_segment / hop_length)
#
#         for s in range(num_segments):
#             start_sample = int(num_samples_per_segment * s)
#             end_sample = int(start_sample + num_samples_per_segment)
#             segment_signal = signal[start_sample:min(end_sample, len(signal))]
#
#             if len(segment_signal) == 0:
#                 print(f"Empty segment signal at segment {s} of file {file_path}. Skipping this segment.")
#                 continue
#
#             feature_vectors = self.extract_feature(segment_signal, sr)
#             feature_vectors = feature_vectors.T
#
#             print(f"The file {filename}, section: {s} is being processed...")
#
#             if len(feature_vectors) == expected_vectors_mfcc:
#                 dict_data[self.feature].append(feature_vectors.tolist())
#                 dict_data["labels"].append(self.treatments.index(treatment))
#                 dict_data["light_treatment"].append(treatment)
#                 dict_data["subfolder_name"].append(self.subfolder)
#                 dict_data["segment_number"].append(s)
#
#         write_gz_json(dict_data, gz_json_path)
#
#         print(f"The file {filename} has been processed.")
#
#     def combine_dicts(self):
#         master_dict = {
#             "light_treatment": [],
#             "mfcc": [],
#             "labels": [],
#             "file_name": [],
#             "segment_number": []
#         }
#
#         for treatment in self.treatments:
#             treatment_path = os.path.join(self.src_directory, treatment)
#             subfolders = sorted([name for name in os.listdir(treatment_path) if os.path.isdir(os.path.join(treatment_path, name))])
#
#             for subfolder in subfolders:
#                 subfolder_path = os.path.join(treatment_path, subfolder)
#
#                 for filename in os.listdir(subfolder_path):
#                     if filename.endswith(".gz"):
#                         gz_json_path = os.path.join(subfolder_path, filename)
#
#                         with gzip.GzipFile(gz_json_path, 'r') as gz_file:
#                             dict_data = json.loads(gz_file.read().decode('utf-8'))
#
#                             master_dict["light_treatment"].extend(dict_data["light_treatment"])
#                             master_dict["mfcc"].extend(dict_data["mfcc"])
#                             master_dict["labels"].extend(dict_data["labels"])
#                             master_dict["file_name"].extend(dict_data["subfolder_name"])
#                             master_dict["segment_number"].extend(dict_data["segment_number"])
#
#         master_file_path = os.path.join(self.dst_directory, "master_file.gz")
#         write_gz_json(master_dict, master_file_path)
#
#     def open_and_view_json(self):
#         master_file_path = os.path.join(self.dst_directory, "master_file.gz")
#         with gzip.GzipFile(master_file_path, 'r') as gz_file:
#             master_dict = json.loads(gz_file.read().decode('utf-8'))
#
#         print(master_dict)
#
# # ------------------------------------------------------
#
# processor = AudioProcessor('G:\Stridulation syllable patterns', 'G:\Test2', 'ae')
# processor.process_single_file('file.wav', 'treatment1')


# class AudioProcessor:
#     def __init__(self, feature: str,
#                  n_mfcc=13, n_fft=2048, hop_length=512, num_segments=30):
#         self.feature = feature  # Feature to extract from audio files
#         self.file_extension = ".gz"
#         self.n_mfcc = n_mfcc
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.num_segments = num_segments
#         self.treatments = []
#         self.features_dir = None
#
#     def create_feature_directory(self, directory, feature):
#         feature_directory = os.path.join(directory, feature)
#         os.makedirs(feature_directory, exist_ok=True)
#         return feature_directory
#
#     def process_single_file(self):
#         root = tk.Tk()
#         root.withdraw()
#         file_path = filedialog.askopenfilename()
#         root.destroy()
#
#         if file_path:
#             filename = os.path.basename(file_path)
#             subfolder_path = os.path.dirname(file_path)
#             treatment = os.path.basename(os.path.dirname(subfolder_path))
#             self.process_file(filename, subfolder_path, treatment)
#         else:
#             print("No file selected.")
#
#     def run(self):
#         src_directory = self.choose_src_directory()
#         dst_directory = self.create_dst_directory()
#         self.features_dir = self.create_feature_directory(dst_directory, self.feature)
#         self.organize_files_by_treatment(src_directory)
#         self.save_features(dst_directory)
#         self.combine_dicts(dst_directory)
#         self.open_and_view_json(dst_directory)
#
#     def choose_src_directory(self):
#         root = tk.Tk()
#         root.withdraw()
#         src_directory = filedialog.askdirectory(title="Select Source Directory")
#         root.destroy()
#         return src_directory
#
#     def create_dst_directory(self):
#         root = tk.Tk()
#         root.withdraw()
#         dst_directory = filedialog.askdirectory(title="Select Destination Directory for Extracted Feature")
#         root.destroy()
#
#         if dst_directory:
#             dst_directory = os.path.join(dst_directory, self.feature)
#             os.makedirs(dst_directory, exist_ok=True)
#
#         return dst_directory
#
#     def organize_files_by_treatment(self, src_directory):
#         assert self.features_dir is not None, "Features directory has not been set."
#         assert src_directory != self.features_dir, "Source and destination directories cannot be the same."
#
#         for treatment in os.listdir(src_directory):
#             treatment_directory = os.path.join(self.features_dir, treatment)
#             os.makedirs(treatment_directory, exist_ok=True)
#
#             treatment_path = os.path.join(src_directory, treatment)
#             for dirpath, _, filenames in os.walk(treatment_path):
#                 for filename in filenames:
#                     if filename.endswith(self.file_extension):
#                         src_file_path = os.path.join(dirpath, filename)
#                         dst_file_path = os.path.join(treatment_directory, filename)
#                         shutil.copy2(src_file_path, dst_file_path)
#
#     def save_features(self, dst_directory):
#         treatments = os.listdir(dst_directory)
#
#         for treatment in treatments:
#             treatment_path = os.path.join(dst_directory, treatment)
#
#             for dirpath, _, filenames in os.walk(treatment_path):
#                 for filename in filenames:
#                     if filename.endswith(".wav"):
#                         self.process_file(filename, dirpath, treatment)
#
#     def extract_feature(self, segment_signal, sr):
#         if self.feature == 'mfcc':
#             return librosa.feature.mfcc(y=segment_signal, sr=sr, n_fft=self.n_fft, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
#         elif self.feature == 'rms':
#             return librosa.feature.rms(y=segment_signal)
#         # elif self.feature == 'ae':
#         #     return librosa.feature.amplitude_envelope(y=segment_signal, sr=sr)
#         else:
#             raise ValueError(f"Invalid feature: {self.feature}")
#
#     def process_file(self, filename, subfolder_path, treatment):
#         file_path = os.path.join(subfolder_path, filename)
#
#         dict_data = {
#             "light_treatment": [],
#             self.feature: [],
#             "labels": [],
#             "subfolder_name": [],
#             "segment_number": []
#         }
#
#         gz_json_file = filename.replace(".wav", f"_{self.feature}.gz")
#         gz_json_path = os.path.join(subfolder_path, gz_json_file)
#         if os.path.isfile(gz_json_path):
#             print(f"'{gz_json_file}' file already exists in {subfolder_path}. Skipping this file.")
#             return
#
#         signal, sr = librosa.load(file_path)
#         duration = librosa.get_duration(y=signal, sr=sr)
#         samples_per_track = sr * duration
#         num_samples_per_segment = samples_per_track / self.num_segments
#         expected_vectors_mfcc = math.ceil(num_samples_per_segment / self.hop_length)
#
#         for s in range(self.num_segments):
#             start_sample = int(num_samples_per_segment * s)
#             end_sample = int(start_sample + num_samples_per_segment)
#             segment_signal = signal[start_sample:end_sample]
#
#             if len(segment_signal) == 0:
#                 print(f"Empty segment signal at segment {s} of file {file_path}. Skipping this segment.")
#                 continue
#
#             feature_vectors = self.extract_feature(segment_signal, sr)
#             feature_vectors = feature_vectors.T
#
#             print(f"The file {filename}, section: {s} is being processed...")
#
#             if len(feature_vectors) == expected_vectors_mfcc:
#                 dict_data[self.feature].append(feature_vectors.tolist())
#                 dict_data["labels"].append(self.treatments.index(treatment))
#                 dict_data["light_treatment"].append(treatment)
#                 dict_data["subfolder_name"].append(subfolder_path)
#                 dict_data["segment_number"].append(s)
#
#         write_gz_json(dict_data, gz_json_path)
#
#         print(f"The file {filename} has been processed.")
#
#     def combine_dicts(self, dst_directory):
#         master_dict = {
#             "light_treatment": [],
#             self.feature: [],
#             "labels": [],
#             "subfolder_name": [],
#             "segment_number": []
#         }
#
#         for treatment in os.listdir(dst_directory):
#             treatment_path = os.path.join(dst_directory, treatment)
#
#             for dirpath, _, filenames in os.walk(treatment_path):
#                 for filename in filenames:
#                     if filename.endswith(".gz"):
#                         gz_json_path = os.path.join(dirpath, filename)
#
#                         with gzip.GzipFile(gz_json_path, 'r') as gz_file:
#                             dict_data = json.loads(gz_file.read().decode('utf-8'))
#
#                             master_dict["light_treatment"].extend(dict_data["light_treatment"])
#                             master_dict[self.feature].extend(dict_data[self.feature])
#                             master_dict["labels"].extend(dict_data["labels"])
#                             master_dict["subfolder_name"].extend(dict_data["subfolder_name"])
#                             master_dict["segment_number"].extend(dict_data["segment_number"])
#
#         master_file_path = os.path.join(dst_directory, "master_file.gz")
#         write_gz_json(master_dict, master_file_path)
#
#     def open_and_view_json(self, dst_directory):
#         master_file_path = os.path.join(dst_directory, "master_file.gz")
#         with gzip.GzipFile(master_file_path, 'r') as gz_file:
#             master_dict = json.loads(gz_file.read().decode('utf-8'))
#
#         print(master_dict)
#
#
# # Example usage
# feature = 'mfcc'
# processor = AudioProcessor(feature)
# processor.process_single_file()
# # processor.run()


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

    def process_file(self, path, filename, subfolder_name, label, segment_number):
        file_path = os.path.join(path, filename)

        dict_data = {
            self.feature: [],
            "labels": [],
            "subfolder_name": [],
            "segment_number": []
        }

        gz_json_file = filename.replace(".wav", f"_{self.feature}.gz")
        treatment_dir = os.path.join(self.features_dir,self.treatments[label])  # Get the corresponding treatment directory
        os.makedirs(treatment_dir, exist_ok=True)  # Create treatment directory if it doesn't exist
        gz_json_path = os.path.join(treatment_dir, gz_json_file)
        if os.path.isfile(gz_json_path):
            print(f"'{gz_json_file}' file already exists in {treatment_dir}. Skipping this file.")
            return

        signal, sr = librosa.load_audio(file_path)
        duration = librosa.get_duration(signal, sr)
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
                dict_data["labels"].append(label)
                dict_data["subfolder_name"].append(subfolder_name)
                dict_data["segment_number"].append(segment_number)

        self.write_gz_json(dict_data, gz_json_path)

        print(f"The file {filename} has been processed.")

    def extract_feature(self, signal, sr):
        if self.feature == "mfcc":
            return librosa.feature.mfcc(signal, sr=sr)
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
feature = 'MFCC'  # Specify the feature to extract
processor = AudioProcessor(feature)
processor.run()
processor.process_single_file()

