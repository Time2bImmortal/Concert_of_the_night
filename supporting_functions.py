import soundfile
import json
import gzip
import matplotlib.pyplot as plt
import librosa
import tkinter as tk
from tkinter import filedialog
import numpy as np


def write_gz_json(json_obj, filename):
    json_str = json.dumps(json_obj) + "\n"
    json_bytes = json_str.encode('utf-8')

    with gzip.GzipFile(filename, 'w') as fout:
        fout.write(json_bytes)


def open_and_show_gz_file():
    """Displays the contents of a gzip file selected by the user.

    This function opens a dialog box to allow the user to select a gzip file (*.gz).
    Once a file is selected, its contents are read and displayed on the console.

    Raises:
        FileNotFoundError: If the selected file is not found.
        gzip.BadGzipFile: If the selected file is not a valid .gz file.
        Exception: If any other error occurs during the file reading process.
       """
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
    """Return the sample rate of an audio file.

    Args:
        audio_file_path (str): The path of the audio file.

    Returns:
        int: The sample rate of the audio file.

    Note:
        The 'soundfile' library is used to read audio files. Make sure it is installed before calling this function.
    """
    data, samplerate = sf.read(audio_file_path)
    print(f'The file has a samplerate of: {samplerate}')
    return samplerate


def display_waveform(signal, sr):
    """ Display the waveform of a signal

    Args:
        signal(ndarray): The signal array
        sr(int): signal sampling rate

    Returns:
        plot

    Note:
        The 'librosa' library and matplotlib.pyplot are used. Make sure they are installed before calling this function.
    """
    plt.figure()
    librosa.display.waveshow(signal, sr=sr)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()


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
