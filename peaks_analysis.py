import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import logging
import os
import shutil


def process_directory(source_root, dest_root, threshold=0.05):
    for dirpath, dirnames, filenames in os.walk(source_root):
        for filename in filenames:
            source_file_path = os.path.join(dirpath, filename)

            if source_file_path.endswith('.wav') and is_valid_audio(source_file_path, threshold):
                # Determine its top-level folder
                relative_path = os.path.relpath(dirpath, source_root)
                top_level_folder = relative_path.split(os.sep)[0]

                dest_folder_path = os.path.join(dest_root, top_level_folder)  # Removed "_filtered" from here

                if not os.path.exists(dest_folder_path):
                    os.makedirs(dest_folder_path)

                dest_file_path = os.path.join(dest_folder_path, filename)
                shutil.copy2(source_file_path, dest_file_path)
def select_audio_file():

    root = Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
    return filepath
def load_audio(filepath):
    logging.info('loading the audio')
    y, sr = librosa.load(filepath, sr=None)  # sr=None ensures original sampling rate is used
    return y, sr


def visualize_waveform(y, sr):
    logging.info('Visualizing the waveform')
    plt.figure(figsize=(6, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    plt.show()

def visualize_spectrogram(y, sr):
    logging.info('Visualizing the spectrogram')
    plt.figure(figsize=(6, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()


def duration_above_amplitude_simple(y, sr, threshold):

    samples_above_threshold = np.sum(np.abs(y) > threshold)

    return samples_above_threshold / sr
def is_valid_audio(file_path, threshold=0.04):
    y, sr = librosa.load(file_path, sr=None)
    return duration_above_amplitude_simple(y, sr, threshold) >= 300

if __name__ == '__main__':
    folder_to_filter = filedialog.askdirectory()
    folder_to_copy = folder_to_filter + '_filtered'
    process_directory(folder_to_filter, folder_to_copy)