import soundfile as sf
import json
import gzip
import matplotlib.pyplot as plt
import librosa
from tkinter import filedialog
from tkinter import Tk
import numpy as np
import os
import shutil
import random
import h5py


def delete_common_files(train_folder, test_folder):
    train_subfolders = get_train_subfolders(train_folder)

    for root, dirs, files in os.walk(test_folder):
        print(root)
        for file in files:
            print(file)
            if file.endswith('.h5'):
                test_subfolder = extract_subfolder_from_filename(file)
                if test_subfolder in train_subfolders:
                    test_file_path = os.path.join(root, file)
                    with gzip.open(test_file_path, 'rt') as f:
                        test_dict_data = json.load(f)
                        test_filename = test_dict_data["filename"]

                    deleted = False
                    for train_root, train_dirs, train_files in os.walk(train_folder):
                        if deleted:
                            break
                        for train_file in train_files:
                            if train_file.endswith('.gz') and extract_subfolder_from_filename(
                                    train_file) == test_subfolder:
                                train_file_path = os.path.join(train_root, train_file)
                                with gzip.open(train_file_path, 'rt') as f:
                                    train_dict_data = json.load(f)
                                    train_filename = train_dict_data["filename"]

                                if test_filename == train_filename:
                                    try:
                                        os.remove(test_file_path)
                                        print(f"Deleted: {test_file_path}")
                                        deleted = True
                                        break
                                    except OSError as e:
                                        print(f"Error deleting {test_file_path}: {e}")


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


def copy_missing_wav_files(treatment_mapping):

    root = Tk()
    root.withdraw()

    # Ask for source and destination folders
    src_folder = filedialog.askdirectory(title="Select Source Folder")
    dest_folder = filedialog.askdirectory(title="Select Destination Folder")

    # Ensure folders exist
    if not os.path.exists(src_folder):
        print(f"Source folder {src_folder} does not exist.")
        return

    if not os.path.exists(dest_folder):
        print(f"Destination folder {dest_folder} does not exist.")
        return

    # Walk through source directory, including subdirectories
    for dirpath, dirnames, filenames in os.walk(src_folder):
        # Get the source subfolder name
        src_subfolder = os.path.basename(dirpath)

        # Determine the treatment based on the source subfolder name
        for key, value in treatment_mapping.items():
            if key in src_subfolder:
                treatment = value
                break
        else:
            continue  # No treatment found for this subfolder, skip

        # Get all .wav files
        wav_files = [f for f in filenames if f.endswith('.wav')]

        # Check if there are any .wav files
        if not wav_files:
            print(f"No .wav files found in source subfolder {src_subfolder}. Skipping this subfolder.")
            continue

        # Compute destination path for current treatment and source subfolder
        dst_dirpath = os.path.join(dest_folder, treatment, src_subfolder)

        # Create directory if it doesn't exist and copy all .wav files
        if not os.path.exists(dst_dirpath):
            os.makedirs(dst_dirpath)
            for wav_file in wav_files:
                shutil.copy2(os.path.join(dirpath, wav_file), dst_dirpath)
                print(wav_file, f"has been copied from {treatment}")
        else:  # Only copy the .wav files that don't already exist in the destination subfolder
            for wav_file in wav_files:
                if not os.path.exists(os.path.join(dst_dirpath, wav_file)):
                    shutil.copy2(os.path.join(dirpath, wav_file), dst_dirpath)
                    print(wav_file, f"has been copied from {treatment}")

    print(f"Copying from {src_folder} to {dest_folder} completed.")


def copy_files_not_in_source(num_files, treatment_mapping):
    """
    Copies a specified number of .wav files from the 'complete' folder to a new 'test' directory.
    The 'test' directory structure is based on the 'source' directory.
    Only files that are not present in the 'source' folder are copied.
    """

    # Create root Tk window and hide it
    root = Tk()
    root.withdraw()

    # Ask for source and complete folders
    source_folder = filedialog.askdirectory(title="Select Source Folder")
    complete_folder = filedialog.askdirectory(title="Select Complete Folder")

    # Create the 'test' directory
    test_folder = os.path.join(os.path.dirname(source_folder), 'test')
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Get a list of all .wav files in the source directory
    source_files = [file for dirpath, _, files in os.walk(source_folder) for file in files if file.endswith('.wav')]

    # Prepare a dictionary to count how many files have been copied for each treatment
    treatment_counter = {treatment: 0 for treatment in treatment_mapping.values()}

    # Iterate over the subdirectories in the complete folder
    for dirpath, dirnames, filenames in os.walk(complete_folder):
        # Get the treatment name from the directory name using the treatment_mapping
        for key, value in treatment_mapping.items():
            if key in dirpath:
                treatment = value
                break
        else:
            continue  # No treatment found for this subfolder, skip

        # Skip this subdirectory if we have already copied enough files for this treatment
        if treatment_counter[treatment] >= num_files:
            continue

        # Get a list of .wav files in the complete subdirectory that are not in the source directory
        new_files = [file for file in filenames if file.endswith('.wav') and file not in source_files]

        # If there are new files, select a number of them randomly
        if new_files:
            selected_files = random.sample(new_files, min(num_files - treatment_counter[treatment], len(new_files)))

            # Update the counter
            treatment_counter[treatment] += len(selected_files)

            # Create the corresponding treatment subdirectory in the 'test' directory and copy the selected files there
            test_subdir = os.path.join(test_folder, treatment, os.path.basename(dirpath))
            if not os.path.exists(test_subdir):
                os.makedirs(test_subdir)
            for file in selected_files:
                shutil.copy2(os.path.join(dirpath, file), os.path.join(test_subdir, file))

    print(f"Copying from {complete_folder} to {test_folder} completed.")


def plot_mfcc_from_h5(sample_rate=44100, frame_size=2048, hop_length=1024):
    frame_length_sec = frame_size / sample_rate
    hop_length_sec = hop_length / sample_rate

    # Using tkinter to create a file dialog
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an .h5 file",
                                           filetypes=(("h5 files", "*.h5"), ("all files", "*.*")))

    if not file_path:
        return

    with h5py.File(file_path, 'r') as f:
        mfccs = f['mel_spectrogram'][:]

    num_segments = mfccs.shape[0]
    num_frames = mfccs.shape[2]
    time_labels = [i * hop_length_sec for i in range(num_frames)]

    for segment_idx in range(num_segments):
        user_input = input(
            f"Press Enter to view MFCC for segment {segment_idx + 1}/{num_segments} or type 'exit' to stop: ")
        if user_input == 'exit':
            break
        plt.imshow(mfccs[segment_idx], cmap='viridis', origin='lower', aspect='auto',
                   extent=[time_labels[0], time_labels[-1], 0, mfccs.shape[1]])
        plt.colorbar()
        plt.title(f"MFCC for Segment {segment_idx + 1}")
        plt.xlabel("Time (s)")
        plt.ylabel("MFCC Coefficients")
        plt.tight_layout()
        plt.show()
def process_treatment(audio_processor, treatment):
    treatment_dir = os.path.join(audio_processor.features_dir, treatment)
    audio_files = audio_processor.audio_files_dict[treatment]
    for file in audio_files:
        print('file:', file)
        audio_processor.process_file(file)


def process_audio_interactive(full=False):
    # Prompt the user to choose a file
    root = Tk()
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


def find_mismatched_h5_files(directory, key, expected_element_shape):
    mismatched_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.h5'):
                file_path = os.path.join(dirpath, filename)
                try:
                    with h5py.File(file_path, 'r') as h5f:
                        if key in h5f:
                            # Check each element in the dataset
                            for element in h5f[key]:
                                if np.shape(element) != expected_element_shape:
                                    mismatched_files.append(file_path)
                                    break  # No need to check other elements for this file
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    return mismatched_files

#
# directory = filedialog.askdirectory()
# key = "mfccs_and_derivatives"
# expected_shape = (15, 862)
# mismatched_files = find_mismatched_h5_files(directory, key, expected_shape)
# print(mismatched_files)
def plot_waveform2(file_path, chirp_positions=None, apply_threshold=False, threshold=0.01):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Create a time axis in seconds
    time_axis = np.linspace(0, len(audio) / sr, num=len(audio))

    # Calculate the amplitude envelope using the Hilbert transform
    amplitude_envelope = np.abs(hilbert(audio))

    # Plot waveform and amplitude envelope
    plt.figure(figsize=(20, 5))
    plt.plot(time_axis, audio, label='Waveform')
    plt.plot(time_axis, amplitude_envelope, color='red', label='Amplitude Envelope')

    # Apply and plot thresholded amplitude envelope if specified
    if apply_threshold:
        amplitude_envelope_thresholded = np.where(amplitude_envelope < threshold, 0, amplitude_envelope)
        plt.plot(time_axis, amplitude_envelope_thresholded, color='green', label='Amplitude Envelope with Threshold')

    # Plot red line at each chirp position
    if chirp_positions is not None:
        for pos in chirp_positions:
            chirp_time = pos / sr  # Convert sample index to time
            plt.axvline(x=chirp_time, color='red', linestyle='--')

    plt.title('Audio Waveform and Amplitude Envelope')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.axhline(0, color='gray', lw=1)
    plt.ylim(-1.0, 1.0)
    plt.legend()
    plt.show()

    def extract_signal_with_threshold(file_path, threshold=None):
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)

        if threshold is not None:
            audio[np.abs(audio) < threshold] = 0

        return audio


def get_wav_file_properties(file_path):
    with wave.open(file_path, 'r') as wav_file:
        num_channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()
        duration = num_frames / float(sample_rate)
        bit_depth = sample_width * 8
        properties = {"filename": file_path.split('/')[-1],
                      "duration": duration,
                      "sample_rate": sample_rate,
                      "bit_depth": bit_depth,
                      "channels": "Mono" if num_channels == 1 else "Stereo"}
        for key, value in properties.items():
            print(f"{key}: {value}")

def add_gaussian_noise_without_saving(mfcc, sigma=0.000):
    noise = torch.randn_like(mfcc) * sigma
    mfcc_with_noise = mfcc + noise
    return mfcc_with_noise


def add_gaussian_noise(mfcc, sigma=0.000):
    # Save the original data to a file (only once)
    np.save('original_data.npy', mfcc.cpu().numpy())

    noise = torch.randn_like(mfcc) * sigma
    mfcc_with_noise = mfcc + noise

    # Save the noisy data to a file (only once)
    np.save('noisy_data.npy', mfcc_with_noise.cpu().numpy())

    return mfcc_with_noise


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# def calculate_cross_correlation(signal1, signal2):
#     # Compute the cross-correlation
#     correlation = np.correlate(signal1, signal2, mode='valid')
#
#     # Normalize the cross-correlation
#     norm = np.linalg.norm(signal1) * np.linalg.norm(signal2)
#     if norm == 0:
#         return np.zeros_like(correlation)
#
#     normalized_correlation = correlation / norm
#     return normalized_correlation

# @print_syllable_details
# def coarse_search(pattern_envelope, signal_envelope, step_size=500, similarity_threshold=0.2):
#     pattern_length = len(pattern_envelope)
#     signal_length = len(signal_envelope)
#
#     if pattern_length > signal_length:
#         raise ValueError("Pattern length cannot be greater than signal length.")
#
#     best_similarity = -1
#     best_position = -1
#
#     for start in range(0, signal_length - pattern_length + 1, step_size):
#         end = start + pattern_length
#         window = signal_envelope[start:end]
#         correlation = calculate_cross_correlation(pattern_envelope, window)
#         max_correlation = np.max(correlation) if len(correlation) > 0 else 0
#
#         if max_correlation > best_similarity:
#             best_similarity = max_correlation
#             best_position = start
#
#         # Stop searching if the similarity threshold is exceeded
#         if max_correlation > similarity_threshold:
#             break
#
#     return best_position, best_similarity


# @print_syllable_details
# def refine_search(pattern_envelope, signal_envelope, initial_position, initial_similarity=None, search_radius=300, step_size=10, non_zero_units_threshold=100):
#     pattern_length = len(pattern_envelope)
#     signal_length = len(signal_envelope)
#
#     if pattern_length > signal_length:
#         raise ValueError("Pattern length cannot be greater than signal length.")
#
#     best_similarity = initial_similarity if initial_similarity is not None else 0
#     best_position = initial_position
#
#     # Find position with best similarity
#     for start in range(max(0, initial_position - search_radius), min(initial_position + search_radius, signal_length - pattern_length), step_size):
#         window = signal_envelope[start:start + pattern_length]
#         correlation = calculate_cross_correlation(pattern_envelope, window)
#         max_correlation = np.max(correlation) if len(correlation) > 0 else 0
#
#         if max_correlation > best_similarity:
#             best_similarity = max_correlation
#             best_position = start
#
#     # Advance the position forward until the next 100 units are all non-zero
#     while best_position + non_zero_units_threshold <= signal_length:
#         post_pattern_region = signal_envelope[best_position:best_position + non_zero_units_threshold]
#         if np.all(post_pattern_region != 0):
#             break  # Found a sequence of 100 non-zero units
#         best_position += 5  # Shift the position forward
#
#     return best_position, best_similarity


# def find_and_analyze_chirps(signal_envelope, syllable_pattern_amplitude, similarity_threshold, step_size):
#
#     pattern_length = len(syllable_pattern_amplitude)
#     signal_length = len(signal_envelope)
#
#     # Initial coarse search to find the first syllable
#     initial_position, initial_similarity = coarse_search(syllable_pattern_amplitude, signal_envelope, step_size, similarity_threshold)
#
#     if initial_similarity <= similarity_threshold:
#         print("No initial syllable found.")
#         return [], [], [], []
#
#     # Refine the initial syllable position
#     refined_initial_position, refined_initial_similarity = refine_search(syllable_pattern_amplitude, signal_envelope, initial_position, initial_similarity)
#
#     syllable_positions = [refined_initial_position]
#
#     current_position = refined_initial_position + int(pattern_length*0.5)
#     while current_position < signal_length - pattern_length:
#
#         search_start = current_position + int(pattern_length*0.5)
#
#         search_end = min(search_start + int(pattern_length * 2), signal_length)
#         if search_end - search_start < pattern_length:
#             break
#
#         coarse_position, coarse_similarity = coarse_search(syllable_pattern_amplitude, signal_envelope[search_start:search_end])
#
#         # Translate the coarse position to the original signal's coordinates
#         absolute_position = search_start + coarse_position
#
#         if coarse_similarity > similarity_threshold:
#             # Refine the search for the next syllable
#             refined_position, refined_similarity = refine_search(syllable_pattern_amplitude, signal_envelope, absolute_position)
#
#             # Append every refined position to syllable_positions
#             syllable_positions.append(refined_position)
#             current_position = refined_position + pattern_length
#         else:
#             current_position += pattern_length
#
#     return syllable_positions
# syllable_pattern = r"C:\Users\yfant\OneDrive\Desktop\Crickets chirps analysis\chirp_LD_main_wave.wav"  # "E:\chirp_LD_main_wave.wav"
# syllable_pattern_amplitude = extract_amplitude_envelope(syllable_pattern)  # hilbert abs
# pattern_length = len(syllable_pattern_amplitude)
# to_erase_files = check_audio_files()
# file_path = select_wav_file()