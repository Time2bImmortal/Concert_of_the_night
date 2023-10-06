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



if __name__ == '__main__':
    Tk().withdraw()
    filename = filedialog.askopenfilename()

    y, sr = librosa.load(filename, sr=None)

    # Select 1/30 of the audio file
    portion_len = len(y) // 30
    y = y[:portion_len]
    print('signal', y.shape)
    # Simple Spectrogram
    plt.figure(figsize=(12, 8))

    D = np.abs(librosa.stft(y))
    print('spectrogram', D.shape)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    plt.title('Spectrogram')
    plt.show()
    # Mel Spectrogram
    plt.subplot(3, 1, 2)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    print('melspec', mel_spec.shape)
    librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel')

    plt.title('Mel Spectrogram')
    plt.show()
    # Mel Spectrogram in dB
    plt.subplot(3, 1, 3)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    print('melspecdb', mel_spec_db.shape)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram (dB)')

    plt.tight_layout()
    plt.show()
