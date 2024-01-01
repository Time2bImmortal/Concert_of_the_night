import wave
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from scipy.signal import hilbert
import soundfile as sf
import csv


""" recommended threshold of 0.005 if needed """

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

def select_wav_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    return file_path

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
def extract_amplitude_envelope(file_path, threshold=None):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)

    amplitude_envelope = np.abs(hilbert(audio))

    if threshold is not None:
        amplitude_envelope[amplitude_envelope < threshold] = 0
    return amplitude_envelope
def extract_signal_with_threshold(file_path, threshold=None):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    if threshold is not None:
        audio[np.abs(audio) < threshold] = 0

    return audio


def calculate_cross_correlation(signal1, signal2):

    # Compute cross-correlation
    correlation = np.correlate(signal2, signal1, mode='valid')

    return correlation
def sliding_window_best_match(pattern_envelope, signal_envelope, step_size=500, similarity_threshold=0.8):
    pattern_length = len(pattern_envelope)
    signal_length = len(signal_envelope)

    if pattern_length > signal_length:
        raise ValueError("Pattern length cannot be greater than signal length.")

    best_similarity = -1
    best_position = -1
    potential_match_found = False
    potential_match_position = -1

    # Initial pass to find a potential match
    for start in range(0, signal_length - pattern_length + 1, step_size):
        window = signal_envelope[start:start + pattern_length]
        correlation = calculate_cross_correlation(pattern_envelope, window)
        max_correlation = np.max(correlation) if len(correlation) > 0 else 0

        if max_correlation > best_similarity:
            best_similarity = max_correlation
            potential_match_position = start
            potential_match_found = True

        if max_correlation > similarity_threshold:
            break  # Stop the initial pass as a potential match is found

    # Refinement phase: search in the vicinity of the potential match
    if potential_match_found:
        for refined_start in range(max(0, potential_match_position - 2 * step_size),
                                   min(potential_match_position + 2 * step_size, signal_length - pattern_length)):
            refined_window = signal_envelope[refined_start:refined_start + pattern_length]
            refined_correlation = calculate_cross_correlation(pattern_envelope, refined_window)
            max_refined_correlation = np.max(refined_correlation) if len(refined_correlation) > 0 else 0

            if max_refined_correlation > best_similarity:
                best_similarity = max_refined_correlation
                best_position = refined_start + np.argmax(refined_correlation)

    return best_position, best_similarity



def find_all_chirps(signal_envelope, pattern_envelope, step_size, similarity_threshold):
    chirp_positions = []
    chirp_groups = []
    current_group = []
    pattern_length = len(pattern_envelope)
    signal_length = len(signal_envelope)

    position = 0
    while position < signal_length - pattern_length:
        best_position, best_similarity = sliding_window_best_match(
            pattern_envelope, signal_envelope[position:], step_size, similarity_threshold)

        if best_similarity > similarity_threshold:
            absolute_position = position + best_position
            chirp_positions.append(absolute_position)

            if current_group and absolute_position - current_group[-1] > 2 * pattern_length:
                chirp_groups.append(current_group)
                current_group = []

            current_group.append(absolute_position)

            # Move to the end of the current chirp to continue searching
            position = absolute_position + pattern_length
        else:
            position += step_size

    if current_group:
        chirp_groups.append(current_group)

    return chirp_positions, chirp_groups

# Function to calculate distances between groups and count chirps in each group
def analyze_chirp_groups(chirp_groups, pattern_length) :
    group_distances = []
    chirp_counts = []

    for i in range(1, len(chirp_groups)):
        distance = chirp_groups[i][0] - chirp_groups[i - 1][-1] + pattern_length
        group_distances.append(distance)

    for group in chirp_groups:
        chirp_counts.append(len(group))

    return group_distances, chirp_counts

def save_chirp_group_order_to_csv(chirp_groups, filename="E:\chirp_group_order.csv"):
    # Prepare chirp group order
    chirp_group_order = [len(group) for group in chirp_groups]

    # Write to CSV
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(chirp_group_order)

def create_modified_audio(file_path, chirp_positions, sample_rate):
    # Load the original audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Create a copy of the audio data
    modified_audio = np.copy(audio)

    # Set the amplitude to 1 at each chirp position
    for pos in chirp_positions:
        if pos < len(modified_audio):
            modified_audio[pos] = 1

    # Define the output file path (e.g., appending "_modified" to the original filename)
    output_path = file_path.replace('.wav', '_modified.wav')

    # Save the modified audio file
    sf.write(output_path, modified_audio, sample_rate)

    print(f"Modified audio file saved as: {output_path}")


def save_results_to_csv(experiment_group, subject, recording_date, total_chirps, chirp_groups, group_distances, average_score, filename="experiment_results.csv"):
    # Prepare chirp group counts
    chirp_group_counts = [0, 0, 0, 0]  # For groups of 2, 3, 4, 5 chirps respectively
    for group in chirp_groups:
        if 2 <= len(group) <= 5:
            chirp_group_counts[len(group) - 2] += 1

    # Prepare data row
    row = [experiment_group, subject, recording_date, total_chirps] + chirp_group_counts + [group_distances, average_score]

    # Write to CSV
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


#################################################################################################
#################################################################################################

"""Use the chirp pattern that you wish, it will run a window to catch similar chirps"""


chirp_pattern = "E:\chirp_LD_main_wave.wav"
chirp_pattern_amplitude = extract_amplitude_envelope(chirp_pattern)
chirp_pattern_signal = extract_signal_with_threshold(chirp_pattern)
print("Pattern size: ", len(chirp_pattern_amplitude))
similarity_threshold = 0.3
step_size = 500  # 50% percent chirp pattern's size
sample_rate = 44100

file_path = select_wav_file()
# select folder/treatment/crickets/files
if file_path:
    # get_wav_file_properties(file_path)
    # plot_waveform2(file_path, apply_threshold=True)
    signal_envelope = extract_amplitude_envelope(file_path, 0.01)
#     signal_envelope = extract_signal_with_threshold(file_path, 0.01)

initial_position, initial_similarity = sliding_window_best_match(chirp_pattern_amplitude, signal_envelope, step_size, similarity_threshold)
print(initial_similarity)
if initial_similarity > similarity_threshold:
    # Start searching for subsequent chirps from the end of the first chirp
    start_position = initial_position + len(chirp_pattern_amplitude)
    chirp_positions, chirp_groups = find_all_chirps(signal_envelope[start_position:], chirp_pattern_amplitude, step_size, similarity_threshold)

    # Adjust positions of found chirps relative to the entire signal
    chirp_positions = [pos + start_position for pos in chirp_positions]
    chirp_positions.insert(0, initial_position)  # Include the initial chirp

    # Determine if the first chirp should be included in the first group
    if chirp_groups and (chirp_groups[0][0] - initial_position) < 1.5 * len(chirp_pattern_amplitude):
        # Include the initial chirp in the first group
        chirp_groups[0].insert(0, initial_position)
    else:
        # Add the initial chirp as its own group
        chirp_groups.insert(0, [initial_position])

    # Analyze chirp groups
    group_distances, chirp_counts = analyze_chirp_groups(chirp_groups, pattern_length=len(chirp_pattern_amplitude))
else:
    print("No initial chirp found.")
if file_path and chirp_positions:
    pass
    # create_modified_audio(file_path, chirp_positions, sample_rate=sample_rate)

    # save_chirp_group_order_to_csv(chirp_groups)

