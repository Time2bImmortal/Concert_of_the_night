import os
import librosa, librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog
import math
SAMPLE_RATE = 44100

#Choose an audio file
root = tk.Tk()
root.withdraw()
dataset_path = filedialog.askdirectory()

# def save_spectrograms(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=30):
#     dict_data = {
#         "light_treatment": [],
#         "mfcc": [],
#         "spectrogram": [],
#         "labels": []
#     }
#
#     for i, (treatment_dirpath, treatment_dirnames, _) in enumerate(os.walk(dataset_path)):
#         if treatment_dirpath is not dataset_path:
#             light_treatment = os.path.basename(treatment_dirpath)
#             dict_data["light_treatment"].append(light_treatment)
#
#             for subject_dirname in treatment_dirnames:
#                 subject_dirpath = os.path.join(treatment_dirpath, subject_dirname)
#
#                 for _, _, filenames in os.walk(subject_dirpath):
#                     for filename in filenames:
#                         file_path = os.path.join(subject_dirpath, filename)
#                         signal, sr = librosa.load(file_path)
#                         duration = librosa.get_duration(path=file_path)
#                         samples_per_track = sr * duration
#                         num_samples_per_segment = samples_per_track / num_segments
#                         expected_vectors_mfcc = math.ceil(num_samples_per_segment / hop_length)
#
#                         for s in range(num_segments):
#                             start_sample = int(num_samples_per_segment * s)
#                             end_sample = int(start_sample + num_samples_per_segment)
#                             segment_signal = signal[start_sample:end_sample]
#                             stft = librosa.core.stft(segment_signal, hop_length=hop_length, n_fft=n_fft)
#                             spectrogram = np.abs(stft)
#                             mfcc = librosa.feature.mfcc(y=segment_signal, sr=sr, n_fft=n_fft, n_mfcc=n_mfcc,
#                                                         hop_length=hop_length)
#                             mfcc = mfcc.T
#                             if len(mfcc) == expected_vectors_mfcc:
#                                 dict_data["mfcc"].append(mfcc.tolist())
#                                 dict_data["labels"].append(i - 1)
#                                 dict_data["spectrogram"].append(spectrogram.tolist())
#                                 print(f"{file_path}, Segment {s}")
#
#     with open(json_path, 'w') as fp:
#         json.dump(dict_data, fp, indent=5)
def save_spectrograms(dataset_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=30):
    treatments = [dir_name for dir_name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, dir_name))]
    for i, (treatment_dirpath, treatment_dirnames, _) in enumerate(os.walk(dataset_path)):
        if treatment_dirpath is not dataset_path:

            light_treatment = os.path.basename(treatment_dirpath)
            label = treatments.index(light_treatment)

            for fdirpath, dirnames, filenames in os.walk(treatment_dirpath):
                for filename in filenames:
                    if not filename.endswith(".wav"):  # Check if the file is an audio file
                        continue

                    dict_data = {
                        "light_treatment": [],
                        "mfcc": [],
                        "spectrogram": [],
                        "labels": [],
                        "subfolder_name": [],
                        "segment_number": []
                    }

                    json_file = filename.replace(".wav", ".json")
                    json_path = os.path.join(fdirpath, json_file)
                    if os.path.isfile(json_path):   # check if json file already exists
                        print(f"'{json_file}' file already exists in {fdirpath}. Skipping this file.")
                        continue

                    file_path = os.path.join(fdirpath, filename)
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

                        stft = librosa.core.stft(segment_signal, hop_length=hop_length, n_fft=n_fft)
                        spectrogram = np.abs(stft)
                        mfcc = librosa.feature.mfcc(y=segment_signal, sr=sr, n_fft=n_fft, n_mfcc=int(n_mfcc), hop_length=hop_length)
                        mfcc = mfcc.T

                        if len(mfcc) == expected_vectors_mfcc:
                            dict_data["mfcc"].append(mfcc.tolist())
                            dict_data["labels"].append(label)
                            dict_data["spectrogram"].append(spectrogram.tolist())
                            dict_data["light_treatment"].append(light_treatment)
                            dict_data["subfolder_name"].append(os.path.basename(fdirpath))  # save the subfolder name
                            dict_data["segment_number"].append(s)

                    with open(json_path, 'w') as fp:
                        json.dump(dict_data, fp, indent=4)


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
    file_path = filedialog.askopenfilename(filetypes=[('JSON Files', '*.json')])

    if file_path:  # if a file was chosen
        with open(file_path, 'r') as fp:
            data_dict = json.load(fp)

        for key, value in data_dict.items():
            print(f"{key}: {value}\n")
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
#

save_spectrograms(dataset_path)