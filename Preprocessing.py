import librosa, librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog
help(librosa.feature.mfcc)
#Choose an audio file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()


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
# samplerate
get_samplerate(file_path)

# waveform
signal, sr = librosa.load(file_path, sr=44100)
# display_waveform(signal, sr)
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))
left_frequency = frequency[:int(len(frequency)/2)]
magnitudeleft = magnitude[:int(len(frequency)/2)]
plt.plot(left_frequency, magnitudeleft)
plt.ylabel("Magnitude")
plt.xlabel("Frequency")
plt.show()

#Frame
n_fft = 2048
hop_length = 512
# stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
# spectogram = np.abs(stft)
# log_spectogram = librosa.amplitude_to_db(spectogram)
# librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

#MFCC
MFCC = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(data=MFCC, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCCs")
plt.colorbar()
plt.show()