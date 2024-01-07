import tkinter as tk
from tkinter import filedialog
# import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from scipy.signal import hilbert
import soundfile as sf
import csv
import os
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
    root = tk.Tk()
    root.withdraw()
    # Ask for source and destination folders
    src_folder = filedialog.askdirectory(title="Select Source Folder")

    file_paths = []

    for dirpath, dirnames, filenames in os.walk(src_folder):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)

    return file_paths
def erase_modified_files():
    root = tk.Tk()
    root.withdraw()
    # Ask for the folder where files need to be erased
    folder = filedialog.askdirectory(title="Select Folder to Clean")

    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if '_modified' in filename:
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

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
        best_position += 5  # Shift the position forward

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
            current_position = refined_position + pattern_length
        else:
            current_position += pattern_length

    return syllable_positions


def find_syllable_ends(signal_envelope, syllable_positions, pattern_length, zero_values_threshold=150, tolerance_percent = 10):
    syllable_end_positions = []
    updated_syllable_start_positions = []
    signal_array = np.array(signal_envelope)

    for position in syllable_positions:
        end_search_start = position + int(pattern_length * 0.3)
        end_search_end = min(position + int(2 * pattern_length), len(signal_array))

        # Only proceed if the segment is long enough for the rolling window
        if end_search_end - end_search_start >= zero_values_threshold:
            windowed_signal = np.lib.stride_tricks.sliding_window_view(signal_array[end_search_start:end_search_end], zero_values_threshold)
            zero_counts = np.sum(windowed_signal == 0, axis=1)

            tolerance = int(zero_values_threshold * tolerance_percent / 100)
            max_non_zeros = zero_values_threshold - tolerance

            valid_windows = np.where(zero_counts >= max_non_zeros)[0]
            if valid_windows.size > 0:
                first_valid_window = valid_windows[0]
                syllable_end_positions.append(end_search_start + first_valid_window)
                updated_syllable_start_positions.append(position)
        # If the segment is too short, skip adding an end position and corresponding start position

    return updated_syllable_start_positions, syllable_end_positions


def group_syllables_and_calculate_distances(syllable_starts, syllable_ends, pattern_length):
    if len(syllable_starts) != len(syllable_ends):
        raise ValueError("The lengths of syllable_starts and syllable_ends do not match.")

    syllable_groups = []
    intra_group_distances = []  # Distances between syllables within groups
    inter_group_distances = []  # Distances between groups

    current_group = []
    for i in range(len(syllable_starts)):
        if not current_group:
            current_group.append((syllable_starts[i], syllable_ends[i]))
        else:
            distance_to_current = syllable_starts[i] - current_group[-1][1]

            if distance_to_current < 2.5 * pattern_length:
                current_group.append((syllable_starts[i], syllable_ends[i]))
                if len(current_group) > 1:
                    intra_group_distances.append(distance_to_current)
            else:
                syllable_groups.append(current_group)
                inter_group_distances.append(distance_to_current)
                current_group = [(syllable_starts[i], syllable_ends[i])]

    if current_group:
        syllable_groups.append(current_group)

    return syllable_groups, intra_group_distances, inter_group_distances


def save_chirp_group_order_to_csv(chirp_groups, filename=r"E:\\chirp_group_order.csv"):
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

def chirps_proportion(chirps_groups):

    chirps_groups_counts = [0,0,0,0,0,0,0,0,0,0,0,0,0]

    for group in chirps_groups:
        chirps_groups_counts[len(group)-1] += 1

    return chirps_groups_counts

def create_csv_from_files(root_folder, csv_filename, syllable_pattern_amplitude, similarity_threshold, step_size, pattern_length):
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["Experiment", "Subject", "File", "Total Syllable Starts", "Total Syllable Ends", "Total Chirps", "Chirp Sizes Proportion", "Mean Inter-Group", "Std Inter-Group", "Mean Intra-Group", "Std Intra-Group"])

        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                parts = file_path.split(os.sep)

                # Extract required details from the path
                experiment = parts[-3]  # Assuming the experiment name is 3 levels up from the file
                subject = parts[-2]  # Assuming the subject name is 2 levels up from the file

                # Process the file
                # Your file processing logic goes here
                signal_envelope = extract_amplitude_envelope(file_path, threshold=0.01)
                syllables_starts = find_and_analyze_chirps(signal_envelope, syllable_pattern_amplitude, similarity_threshold, step_size)
                syllable_ends = find_syllable_ends(signal_envelope, syllables_starts, pattern_length)
                chirps, intra_distances, inter_distances = group_syllables_and_calculate_distances(syllables_starts, syllable_ends, pattern_length)
                chirps_proportions = chirps_proportion(chirps)

                syllables_total_number = len(syllables_starts)
                syllable_ends_number = len(syllable_ends)
                chirps_number = len(chirps)
                intra_mean = round(np.mean(intra_distances), 1) if intra_distances else 0
                intra_std = round(np.std(intra_distances), 1) if intra_distances else 0
                inter_mean = round(np.mean(inter_distances), 1) if inter_distances else 0
                inter_std = round(np.std(inter_distances), 1) if inter_distances else 0

                # Write the data row
                writer.writerow([experiment, subject, filename, syllables_total_number, syllable_ends_number, chirps_number, chirps_proportions, intra_mean, intra_std, inter_mean, inter_std])

def match_starts_and_ends(syllables_starts, syllables_ends):

    if syllables_ends and syllables_starts and syllables_ends[0] < syllables_starts[0]:
        syllables_ends.pop(0)

    if syllables_starts and syllables_ends and syllables_starts[-1] > syllables_ends[-1]:
        syllables_starts.pop()

    if len(syllables_starts) != len(syllables_ends):
        idx = 0
        while idx < min(len(syllables_starts), len(syllables_ends)):
            # If a start time is greater than its corresponding end time, it's a mismatch
            if syllables_starts[idx] > syllables_ends[idx]:
                # Identify whether to remove a start or an end based on which list is longer
                if len(syllables_starts) > len(syllables_ends):
                    mismatched_time = syllables_starts.pop(idx)
                    mismatch_type = "start"
                else:
                    mismatched_time = syllables_ends.pop(idx)
                    mismatch_type = "end"
            print("******************************************************************************")
            print(f"Mismatch eliminated at position {idx}: {mismatch_type} time {mismatched_time}")
            print("******************************************************************************")
        else:
            idx += 1

    return syllables_starts, syllables_ends


def preprocess_amplitude_if_needed(signal, default_threshold=0.04):
    signal = np.array(signal)

    # Condition: Check if all values are below 0.1
    if np.all(signal < 0.1):
        modified_signal = signal * 9
        modified_signal[np.abs(modified_signal) < 0.08] = 0
        return modified_signal

    # Condition: Check if all values are below 0.3
    elif np.all(signal < 0.3):
        modified_signal = signal * 3
        modified_signal[np.abs(modified_signal) < 0.08] = 0
        return modified_signal

    # Condition: If there are more than 1000 ones
    count_of_ones = np.sum(signal == 1)
    if count_of_ones >= 1:
        reduced_signal = signal / 2
        reduced_signal[np.abs(reduced_signal) < 0.05] = 0
        return reduced_signal

    # Default case: Apply a threshold of 0.03
    signal[np.abs(signal) < default_threshold] = 0
    return signal

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
csv_filename = r'E:\\results_chirps_analysis.csv'
bugged_files = []

# file_path = select_wav_file()
erase_modified_files()
files_paths = select_folder_experiment()
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["Experiment", "Subject", "File", "Total Syllable Starts", "Total Syllable Ends", "Total Chirps",
                     "Chirp Sizes Proportion", "Mean Inter-Group", "Std Inter-Group", "Mean Intra-Group",
                     "Std Intra-Group"])

    for file_path in files_paths:
        if file_path is None:
            continue
        print('Currently processing :', file_path)

        parts = file_path.split(os.sep)

        # Extract required details from the path
        experiment = parts[-3]  # Assuming the experiment name is 3 levels up from the file
        subject = parts[-2]  # Assuming the subject name is 2 levels up from the file


        signal_envelope = extract_amplitude_envelope(file_path)  # hilbert abs / threshold improvement
        signal_envelope = preprocess_amplitude_if_needed(signal_envelope)
        signal_length = len(signal_envelope)

        syllables_starts = find_and_analyze_chirps(signal_envelope, syllable_pattern_amplitude, similarity_threshold, step_size)
        syllable_starts, syllable_ends = find_syllable_ends(signal_envelope, syllables_starts, pattern_length)

        # Handle partial pattern at the extremities
        if len(syllables_starts) != len(syllable_ends):
            syllables_starts, syllable_ends = match_starts_and_ends(syllables_starts, syllable_ends)

        chirps, intra_distances, inter_distances = group_syllables_and_calculate_distances(syllables_starts, syllable_ends, pattern_length)
        chirps_proportion_result = chirps_proportion(chirps)

        # So now we have the syllables start and end and the chirps list.
        syllables_total_number = len(syllables_starts)
        syllable_ends_number = len(syllable_ends)
        chirps_number = len(chirps)
        print('Syllables starts total number =', syllables_total_number)
        print('Syllables ends total number =', syllable_ends_number)
        print('Chirps total number =', chirps_number)
        intra_mean = np.mean(intra_distances) if intra_distances else 0
        intra_std = np.std(intra_distances) if intra_distances else 0

        # Calculate mean and standard deviation for intergroup distances
        inter_mean = np.mean(inter_distances) if inter_distances else 0
        inter_std = np.std(inter_distances) if inter_distances else 0

        # print("Intra-group Distances: Mean =", round(intra_mean, 1), ", Standard Deviation =", round(intra_std, 1))
        # print("Inter-group Distances: Mean =", round(inter_mean, 1), ", Standard Deviation =", round(inter_std, 1))
        writer.writerow([experiment, subject, os.path.basename(file_path), syllables_total_number, syllable_ends_number,
                         chirps_number, chirps_proportion_result, intra_mean, intra_std, inter_mean, inter_std])
        create_modified_audio(file_path, syllables_starts, syllable_ends, 44100)


