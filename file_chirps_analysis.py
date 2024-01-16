import tkinter as tk
from tkinter import filedialog
# import matplotlib.pyplot as plt
import librosa
# import librosa.display
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
            if '_modified' in filename or '_labels' in filename:
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
def check_audio_files():
    root = tk.Tk()
    root.withdraw()
    # Ask for the folder where files need to be checked
    folder = filedialog.askdirectory(title="Select Folder to Check")
    below_files=[]
    threshold = 0.2  # Set your desired threshold here
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith('.wav'):  # Check if the file is an audio file
                file_path = os.path.join(dirpath, filename)
                try:
                    audio, sr = librosa.load(file_path, sr=None)  # Load the audio file
                    if np.max(np.abs(audio)) < threshold:  # Check if any value exceeds the threshold
                        print(dirpath, filename)
                        below_files.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return below_files
def extract_amplitude_envelope(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=44100)
        dominant_frequency = extract_dominant_frequency(audio)
        audio = audio / np.max(np.abs(audio))
        amplitude_envelope = np.abs(hilbert(audio))

        return amplitude_envelope, dominant_frequency

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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

def find_and_analyze_chirps(signal_envelope, amplitude_threshold):
    window_size = 100
    step_size = 100
    continuous_threshold = 100
    syllable_length = 800  # Average length of a syllable
    zero_threshold = 100
    zero_tolerance = 5
    syllable_positions = []
    penalty_threshold = 50  # Threshold for the penalty count
    post_check_length = 500

    start = 0
    penalty_count=0

    while start <= len(signal_envelope) - window_size:
        window = signal_envelope[start:start + window_size]
        above_threshold = window > amplitude_threshold
        above_threshold_counts = np.sum(above_threshold)

        if above_threshold_counts >= continuous_threshold:
            refined_position = refine_position(signal_envelope, start, zero_threshold, zero_tolerance)
            if refined_position != -1:

                syllable_positions.append(refined_position)
                start = refined_position + syllable_length  # Skip forward
                continue

        start += step_size

    return syllable_positions  # In case no valid positions are found


def refine_position(signal, start_position, zero_threshold, zero_tolerance):
    max_check_length = 200
    end_check = max(0, start_position - max_check_length)
    segment = signal[end_check:start_position]

    if segment.size < zero_threshold:
        return -1  # Segment is too short to contain the required number of zeros

    segment_reversed = segment[::-1]
    windowed_signal = np.lib.stride_tricks.sliding_window_view(segment_reversed, zero_threshold)
    zero_counts = np.sum(windowed_signal == 0, axis=1)
    max_non_zeros = zero_threshold - zero_tolerance

    valid_windows = np.where(zero_counts >= max_non_zeros)[0]
    if valid_windows.size > 0:
        first_valid_window = valid_windows[0]
        # Translate the position back to the original signal orientation
        return start_position - first_valid_window - zero_threshold

    return -1  # If the condition is not met in any segment


# Example usage:
# refined_position = refine_position(signal_envelope, start_position, 100, 5)


def find_syllable_ends(signal_envelope, syllable_positions, zero_values_threshold=100, tolerance_percent = 5):
    syllable_end_positions = []
    signal_array = np.array(signal_envelope)

    for position in syllable_positions:
        end_search_start = position + int(600)
        end_search_end = min(position + int(1400), len(signal_array))

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
    return syllable_end_positions


def group_syllables_and_calculate_distances(syllable_starts, syllable_ends):
    if len(syllable_starts) != len(syllable_ends):
        raise ValueError("The lengths of syllable_starts and syllable_ends do not match.")

    syllable_groups = []
    intra_group_distances = []  # Distances between syllables within groups
    inter_group_distances = []  # Distances between groups
    group_size = []
    current_group = []
    for i in range(len(syllable_starts)):

        if not current_group:
            current_group.append((syllable_starts[i], syllable_ends[i]))
        else:
            distance_to_current = syllable_starts[i] - current_group[-1][1]

            if distance_to_current < 2400:
                if len(current_group) > 1:
                    intra_group_distances.append(distance_to_current)
                current_group.append((syllable_starts[i], syllable_ends[i]))

            else:
                group_start = current_group[0][0]
                group_end = current_group[-1][1]
                middle_position = (group_start + group_end) // 2
                group_size.append(group_end-group_start)
                syllable_groups.append((middle_position, len(current_group)))

                current_group = [(syllable_starts[i], syllable_ends[i])]
                if i < len(syllable_starts) - 1:
                    inter_group_distances.append(distance_to_current)

    # Handle the last group if it exists
    if current_group:
        group_start = current_group[0][0]
        group_end = current_group[-1][1]
        middle_position = (group_start + group_end) // 2
        syllable_groups.append((middle_position, len(current_group)))

    return syllable_groups, intra_group_distances, inter_group_distances, group_size


def create_modified_audio(file_path, syllables_positions, syllables_ends, chirps_groups):
    # Load the original audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Create a copy of the audio data
    modified_audio = np.copy(audio)

    # Set the amplitude at each syllable position
    for pos in syllables_positions:
        if pos < len(modified_audio):
            modified_audio[int(pos)] = 1
    for pos in syllables_ends:
        if pos < len(modified_audio):
            modified_audio[int(pos)] = -1

    # Define the output file paths
    output_audio_path = file_path.replace('.wav', '_modified.wav')
    output_label_path = file_path.replace('.wav', '_labels.txt')

    # Save the modified audio file
    sf.write(output_audio_path, modified_audio, 44100)

    # # Create and save the label track file
    with open(output_label_path, 'w') as label_file:
        for middle_position, group_size in chirps_groups:
            # Convert the middle position to time in seconds
            time_position = middle_position / 44100
            label_file.write(f"{time_position}\t{time_position}\tGroup Size: {group_size}\n")

    print(f"Modified audio file saved as: {output_audio_path}")
    print(f"Label track file saved as: {output_label_path}")

def chirps_proportion(chirps_groups):
    chirps_groups_counts = {}

    for _, group_size in chirps_groups:
        # Increment the count for the group size, initializing if not present
        if group_size in chirps_groups_counts:
            chirps_groups_counts[group_size] += 1
        else:
            chirps_groups_counts[group_size] = 1

    # Determine the maximum group size dynamically
    max_group_size = max(chirps_groups_counts.keys(), default=0)
    counts_list = [chirps_groups_counts.get(i, 0) for i in range(1, max_group_size + 1)]

    return counts_list


def match_starts_and_ends(syllables_starts, syllables_ends):
    mismatch=0
    # Remove leading 'end' if it comes before the first 'start'
    while syllables_ends and syllables_starts and syllables_ends[0] < syllables_starts[0]:
        syllables_ends.pop(0)

        mismatch+=1
    # Remove trailing 'start' if it's after the last 'end'
    while syllables_starts and syllables_ends and syllables_starts[-1] > syllables_ends[-1]:
        syllables_starts.pop()
        mismatch += 1

    paired_starts = []
    paired_ends = []

    start_idx, end_idx = 0, 0

    while start_idx < len(syllables_starts) and end_idx < len(syllables_ends):
        start = syllables_starts[start_idx]

        # Find the nearest end after the current start
        while end_idx < len(syllables_ends) and syllables_ends[end_idx] < start:
            end_idx += 1

        # Check if we have run out of ends or if the next start comes before the current end
        if end_idx == len(syllables_ends) or (
                start_idx + 1 < len(syllables_starts) and syllables_starts[start_idx + 1] < syllables_ends[end_idx]):
            start_idx += 1
            continue

        end = syllables_ends[end_idx]

        paired_starts.append(start)
        paired_ends.append(end)

        # Move to the next valid start
        next_start_idx = start_idx + 1
        while next_start_idx < len(syllables_starts) and syllables_starts[next_start_idx] - start < 400:
            next_start_idx += 1

        start_idx = next_start_idx
        end_idx += 1

    return paired_starts, paired_ends, mismatch


def preprocess_amplitude_if_needed(signal, default_threshold=0.08):
    if signal is None:
        print("Received None as input for preprocessing. Exiting function.")
        return None
    signal = np.array(signal)

    # Normalize the signal to peak at 0.5
    max_value = np.max(signal)  # Find the maximum value
    if max_value > 0:  # Ensure max_value is not zero to avoid division by zero
        signal = 0.9 * signal / max_value

    # Apply a threshold to reduce low amplitude noise
    # Adjust the threshold as needed for your specific application
    signal[signal < default_threshold] = 0

    return signal


def mean_syllable_size(syllable_starts, syllable_ends):
    if len(syllable_starts) != len(syllable_ends):
        raise ValueError("The lengths of syllable_starts and syllable_ends lists must be the same.")

    # Calculate the differences and check for negative sizes
    differences = []
    for start, end in zip(syllable_starts, syllable_ends):
        size = end - start
        if size < 0:
            raise ValueError(f"Negative syllable size detected: start {start}, end {end}")
        differences.append(size)

    # Calculate the mean size
    mean_size = sum(differences) / len(differences)
    return mean_size
def mean_group_size(group):
    mean_size = sum(group) / len(group)
    return mean_size

def extract_dominant_frequency(signal, sampling_rate=44100):
    # Apply FFT to the signal
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_result), 1/sampling_rate)

    # Get the absolute values to find the magnitude
    magnitude = np.abs(fft_result)

    # Find the index of the maximum magnitude
    dominant_index = np.argmax(magnitude)

    # Find the dominant frequency
    dominant_frequency = frequencies[dominant_index]

    return dominant_frequency
#################################################################################################
#################################################################################################
#################################################################################################

"""Use the chirp pattern that you wish, it will run a window to catch similar chirps"""

print("The script is starting...")

syllable_pattern = r"C:\Users\yfant\OneDrive\Desktop\Crickets chirps analysis\chirp_LD_main_wave.wav"  # "E:\chirp_LD_main_wave.wav"
syllable_pattern_amplitude = extract_amplitude_envelope(syllable_pattern)  # hilbert abs
similarity_threshold = 0.3
step_size = 600  # 50% percent chirp pattern's size
sample_rate = 44100
mismatch=0
pattern_length = len(syllable_pattern_amplitude)
csv_filename = r'D:\new_test.csv'
bugged_files = []

# to_erase_files = check_audio_files()
# file_path = select_wav_file()
erase_modified_files()

files_paths = select_folder_experiment()
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["Experiment", "Subject", "File", "Total Syllable Starts", "Total Syllable Ends", "Total Chirps",
                     "Chirp Sizes Proportion", "Mean Inter-Group", "Std Inter-Group", "Mean Intra-Group",
                     "Std Intra-Group", "Mean syllable size", "Mean chirp size", "Dominant frequency"])

    for file_path in files_paths:
        if file_path is None:
            continue
        print('Currently processing :', file_path)

        parts = file_path.split(os.sep)

        # Extract required details from the path
        experiment = parts[-3]  # Assuming the experiment name is 3 lenvels up from the file
        subject = parts[-2]  # Assuming the subject name is 2 levels up from the file

        signal_envelope, dominant_frequency = extract_amplitude_envelope(file_path)  # hilbert abs / threshold improvement
        signal_envelope = preprocess_amplitude_if_needed(signal_envelope)
        signal_length = len(signal_envelope)

        # syllables_starts = find_and_analyze_chirps(signal_envelope, syllable_pattern_amplitude, similarity_threshold, step_size)
        syllables_starts = find_and_analyze_chirps(signal_envelope, amplitude_threshold=0.08)


        syllable_ends = find_syllable_ends(signal_envelope, syllables_starts)
        syllables_starts, syllable_ends, mismatch = match_starts_and_ends(syllables_starts, syllable_ends)
        mean_size_syllable = mean_syllable_size(syllables_starts, syllable_ends)

        chirps, intra_distances, inter_distances, group_size = group_syllables_and_calculate_distances(syllables_starts,syllable_ends)
        mean_chirp_size = mean_group_size(group_size)
        chirps_proportion_result = chirps_proportion(chirps)

        print('Syllables starts total number =', len(syllables_starts))
        print('Syllables ends total number =', len(syllable_ends))
        print('Chirps total number =', len(chirps))
        intra_mean = np.mean(intra_distances) if intra_distances else 0
        intra_std = np.std(intra_distances) if intra_distances else 0

        # Calculate mean and standard deviation for intergroup distances
        inter_mean = np.mean(inter_distances) if inter_distances else 0
        inter_std = np.std(inter_distances) if inter_distances else 0

        # print("Intra-group Distances: Mean =", round(intra_mean, 1), ", Standard Deviation =", round(intra_std, 1))
        # print("Inter-group Distances: Mean =", round(inter_mean, 1), ", Standard Deviation =", round(inter_std, 1))
        writer.writerow([experiment, subject, os.path.basename(file_path), len(syllables_starts), len(syllable_ends),
                         len(chirps), chirps_proportion_result, round(intra_mean,0), round(intra_std,0),
                         round(inter_mean,0), round(inter_std,0), round(mean_size_syllable,0), round(mean_chirp_size,0), dominant_frequency])
        create_modified_audio(file_path, syllables_starts, syllable_ends, chirps)


