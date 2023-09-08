import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import logging
import os
import shutil
import matplotlib.pyplot as plt

def process_directory(source_root, dest_root, threshold=0.01):
    for dirpath, dirnames, filenames in os.walk(source_root):
        for filename in filenames:
            logging.info(f'Process {filename}')
            source_file_path = os.path.join(dirpath, filename)

            if source_file_path.endswith('.wav') and is_valid_audio(source_file_path, threshold):
                # Determine its top-level folder
                relative_path = os.path.relpath(dirpath, source_root)
                top_level_folder = relative_path.split(os.sep)[0]

                dest_folder_path = os.path.join(dest_root, top_level_folder)  # Removed "_filtered" from here

                if not os.path.exists(dest_folder_path):
                    os.makedirs(dest_folder_path)

                dest_file_path = os.path.join(dest_folder_path, filename)

                # Check if the file already exists in the destination path
                if os.path.exists(dest_file_path):
                    logging.info(f'{filename} already exists in the destination. Skipping.')
                    continue

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


def visualize_mfcc(y, sr, n_mfcc=13):
    plt.figure(figsize=(25, 10))

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfccs = librosa.feature.delta(mfccs)
    # delta_mfccs2 = librosa.feature.delta(mfccs, order=2)

    # Concatenate MFCCs and derivatives
    # combined_mfccs = np.concatenate([mfccs, delta_mfccs, delta_mfccs2])

    # Display the concatenated features
    librosa.display.specshow(delta_mfccs, sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.title('delta mfcc')

    # Set y-axis to differentiate MFCCs, Delta, and Delta-Delta
    # yticks_positions = np.arange(0, 3 * n_mfcc, n_mfcc) + n_mfcc / 2
    # yticks_labels = ['MFCC', 'Delta', 'Delta-Delta']

    plt.tight_layout()
    plt.show()


def check_file_size(file_path, min_size=0, max_size=float('inf')):
    """Check if file size is within given limits."""
    file_size = os.path.getsize(file_path)
    return min_size <= file_size <= max_size


def duration_above_amplitude_simple(y, sr, threshold):
    """Calculate duration of samples above a given threshold."""
    samples_above_threshold = np.sum(np.abs(y) > threshold)
    return samples_above_threshold / sr


def is_valid_audio(file_path, threshold=0.01):
    """Check if audio file has sufficient duration above given amplitude threshold."""
    y, sr = librosa.load(file_path, sr=None)
    return duration_above_amplitude_simple(y, sr, threshold) >= 300


def find_valid_folders(src_directory, min_size=0, max_size=float('inf')):
    """Find folders with at least 20 valid .wav files and write their paths to a text file."""

    output_file = os.path.join(os.path.dirname(src_directory), "valid_folders.txt")

    with open(output_file, 'w') as file:
        for root, _, files in os.walk(src_directory):
            valid_files = [os.path.join(root, file_name) for file_name in files if
                           file_name.lower().endswith('.wav') and
                           check_file_size(os.path.join(root, file_name), min_size, max_size) and
                           is_valid_audio(os.path.join(root, file_name))]

            if len(valid_files) >= 20:
                file.write(root + '\n')
                for valid_file in valid_files:
                    file.write(valid_file + '\n')
                file.write('\n')  # Add an extra newline for clarity

    print(f"Paths written to {output_file}.")


def select_folder_and_process():
    """Use a GUI dialog to select source folder and process."""
    root = Tk()
    root.withdraw()  # Hide the main window
    src_directory = filedialog.askdirectory(title="Choose source folder")
    if src_directory:
        find_valid_folders(src_directory, min_size=1024, max_size=10 * 1024 * 1024)  # example size range 1KB to 10MB
    else:
        print("No folder selected.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # folder_to_filter = filedialog.askdirectory()
    # folder_to_copy = folder_to_filter + '_filtered'
    # process_directory(folder_to_filter, folder_to_copy)
    file_to_analyze = filedialog.askopenfilename()

    signal, sr = load_audio(file_to_analyze)
    visualize_mfcc(signal, sr)
