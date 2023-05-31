import os

def plot_sound(filename):
    # Load audio file
    print(os.path.exists(filename))
    # Load a .wav file
    sr, audio = wavfile.read(filename)

    # If stereo, convert to mono by averaging the two channels
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Plot waveform
    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    plt.title('Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.show()

    # Plot spectrogram
    plt.figure(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()


def list_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        return files
    except FileNotFoundError:
        print("The specified directory does not exist.")
        return []
    except Exception as e:
        print("An error occurred while accessing the directory.")
        print(str(e))
        return []

