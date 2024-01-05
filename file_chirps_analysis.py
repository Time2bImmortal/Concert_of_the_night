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

def print_syllable_details(func):
    def wrapper(*args, **kwargs):
        syllable_position, similarity = func(*args, **kwargs)
        time_position = syllable_position / 44100  # Assuming a sample rate of 44100 Hz

        print(f"Syllable Position: {syllable_position} (Sample Number)")
        print(f"Time Position: {time_position:.2f} seconds")
        print(f"Similarity: {similarity}")

        return syllable_position, similarity
    return wrapper
def select_wav_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    return file_path


def select_folder_experiment():
    pass


def extract_amplitude_envelope(file_path, threshold=None):

    audio, sr = librosa.load(file_path, sr=None)

    amplitude_envelope = np.abs(hilbert(audio))

    if threshold is not None:
        amplitude_envelope[amplitude_envelope < threshold] = 0
    return amplitude_envelope


def calculate_cross_correlation(signal1, signal2):
    # Compute the cross-correlation
    correlation = np.correlate(signal1, signal2, mode='valid')

    # Normalize the cross-correlation
    norm = np.linalg.norm(signal1) * np.linalg.norm(signal2)
    if norm == 0:
        return np.zeros_like(correlation)

    normalized_correlation = correlation / norm
    return normalized_correlation

# @print_syllable_details
def coarse_search(pattern_envelope, signal_envelope, step_size=500, similarity_threshold=0.2):
    pattern_length = len(pattern_envelope)
    signal_length = len(signal_envelope)

    if pattern_length > signal_length:
        raise ValueError("Pattern length cannot be greater than signal length.")

    best_similarity = -1
    best_position = -1

    for start in range(0, signal_length - pattern_length + 1, step_size):
        end = start + pattern_length
        window = signal_envelope[start:end]
        correlation = calculate_cross_correlation(pattern_envelope, window)
        max_correlation = np.max(correlation) if len(correlation) > 0 else 0

        if max_correlation > best_similarity:
            best_similarity = max_correlation
            best_position = start

        # Stop searching if the similarity threshold is exceeded
        if max_correlation > similarity_threshold:
            break

    return best_position, best_similarity


# @print_syllable_details
def refine_search(pattern_envelope, signal_envelope, initial_position, initial_similarity=None, search_radius=300, step_size=10, non_zero_units_threshold=100):
    pattern_length = len(pattern_envelope)
    signal_length = len(signal_envelope)

    if pattern_length > signal_length:
        raise ValueError("Pattern length cannot be greater than signal length.")

    best_similarity = initial_similarity if initial_similarity is not None else 0
    best_position = initial_position

    # Find position with best similarity
    for start in range(max(0, initial_position - search_radius), min(initial_position + search_radius, signal_length - pattern_length), step_size):
        window = signal_envelope[start:start + pattern_length]
        correlation = calculate_cross_correlation(pattern_envelope, window)
        max_correlation = np.max(correlation) if len(correlation) > 0 else 0

        if max_correlation > best_similarity:
            best_similarity = max_correlation
            best_position = start

    # Advance the position forward until the next 100 units are all non-zero
    while best_position + non_zero_units_threshold <= signal_length:
        post_pattern_region = signal_envelope[best_position:best_position + non_zero_units_threshold]
        if np.all(post_pattern_region != 0):
            break  # Found a sequence of 100 non-zero units
        best_position += 10  # Shift the position forward

    return best_position, best_similarity


def find_and_analyze_chirps(signal_envelope, syllable_pattern_amplitude, similarity_threshold, step_size):

    pattern_length = len(syllable_pattern_amplitude)
    signal_length = len(signal_envelope)

    # Initial coarse search to find the first syllable
    initial_position, initial_similarity = coarse_search(syllable_pattern_amplitude, signal_envelope, step_size, similarity_threshold)

    if initial_similarity <= similarity_threshold:
        print("No initial syllable found.")
        return [], [], [], []

    # Refine the initial syllable position
    refined_initial_position, refined_initial_similarity = refine_search(syllable_pattern_amplitude, signal_envelope, initial_position, initial_similarity)

    syllable_positions = [refined_initial_position]
    syllable_groups = [[refined_initial_position]]

    current_position = refined_initial_position + int(pattern_length*0.4)
    while current_position < signal_length - pattern_length:
        search_start = current_position + int(pattern_length*0.4)

        search_end = min(search_start + int(pattern_length * 3), signal_length)
        if search_end - search_start < pattern_length:
            break

        coarse_position, coarse_similarity = coarse_search(syllable_pattern_amplitude, signal_envelope[search_start:search_end])

        # Translate the coarse position to the original signal's coordinates
        absolute_position = search_start + coarse_position

        if coarse_similarity > similarity_threshold:
            # Refine the search for the next syllable
            refined_position, refined_similarity = refine_search(syllable_pattern_amplitude, signal_envelope, absolute_position)

            # Append every refined position to syllable_positions
            syllable_positions.append(refined_position)

            # Add to the current group or start a new group
            if syllable_groups[-1] and (refined_position - syllable_groups[-1][-1]) < 1.5 * pattern_length:
                syllable_groups[-1].append(refined_position)
            else:
                syllable_groups.append([refined_position])

            current_position = refined_position + pattern_length
        else:
            current_position += pattern_length

    return syllable_positions, syllable_groups


def find_syllable_ends(signal_envelope, syllable_positions, pattern_length, zero_values_threshold=100, tolerance_percent=10):
    syllable_end_positions = []
    signal_array = np.array(signal_envelope)

    for position in syllable_positions:
        end_search_start = position + int(pattern_length*0.5)
        end_search_end = min(position + int(1.5 * pattern_length), len(signal_array))

        # Generate a rolling window of size zero_values_threshold
        windowed_signal = np.lib.stride_tricks.sliding_window_view(signal_array[end_search_start:end_search_end], zero_values_threshold)
        zero_counts = np.sum(windowed_signal == 0, axis=1)

        # Calculate the allowed number of non-zero values within the window
        tolerance = int(zero_values_threshold * tolerance_percent / 100)
        max_non_zeros = zero_values_threshold - tolerance

        # Find the first window where non-zero count is within tolerance
        valid_windows = np.where(zero_counts >= max_non_zeros)[0]
        if valid_windows.size > 0:
            first_valid_window = valid_windows[0]
            syllable_end_positions.append(end_search_start + first_valid_window)

    return syllable_end_positions



def analyze_chirp_groups(chirp_groups) :
    chirp_counts = []

    for group in chirp_groups:
        chirp_counts.append(len(group))

    return chirp_counts


def save_chirp_group_order_to_csv(chirp_groups, filename=r"E:\\"):
    # Prepare chirp group order
    chirp_group_order = [len(group) for group in chirp_groups]

    # Write to CSV
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(chirp_group_order)


def create_modified_audio(file_path, syllables_positions, syllables_ends, sample_rate):
    # Load the original audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Create a copy of the audio data
    modified_audio = np.copy(audio)

    # Set the amplitude to 1 at each chirp position
    for pos in syllables_positions:
        if pos < len(modified_audio):
            modified_audio[int(pos)] = 1
    for pos in syllables_ends:
        if pos < len(modified_audio):
            modified_audio[int(pos)] = -1

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
#################################################################################################

"""Use the chirp pattern that you wish, it will run a window to catch similar chirps"""

print("The script is starting...")

syllable_pattern = r"E:\chirp_LD_main_wave.wav"  # "E:\chirp_LD_main_wave.wav"
syllable_pattern_amplitude = extract_amplitude_envelope(syllable_pattern)  # hilbert abs
similarity_threshold = 0.3
step_size = 600  # 50% percent chirp pattern's size
sample_rate = 44100
pattern_length = len(syllable_pattern_amplitude)



file_path = select_wav_file()
# files_paths = select_folder_experiment()

if file_path:

    signal_envelope = extract_amplitude_envelope(file_path, threshold=0.01)  # hilbert abs / threshold improvement
    signal_length = len(signal_envelope)

    syllables_positions, chirps_list = find_and_analyze_chirps(signal_envelope, syllable_pattern_amplitude, similarity_threshold, step_size)
    syllable_ends = find_syllable_ends(signal_envelope, syllables_positions, pattern_length)
    # So now we have the syllables start and end, the chirps list.
    syllables_total_number = len(syllables_positions)
    syllable_ends_number = len(syllable_ends)


if file_path and syllables_positions:
    create_modified_audio(file_path, syllables_positions, syllable_ends, sample_rate=sample_rate)
    save_chirp_group_order_to_csv(chirps_list)

