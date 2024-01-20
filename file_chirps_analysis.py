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


def process_audio(file_path, smoothing_window=20, scaling_factor=0.9, threshold=0.08):
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=44100)
        # Normalize the audio
        audio = audio / np.max(np.abs(audio))

        # Extract the amplitude envelope
        amplitude_envelope = np.abs(hilbert(audio))

        # Smooth the amplitude envelope
        smoothed_amplitude_envelope = np.convolve(amplitude_envelope, np.ones(smoothing_window) / smoothing_window, mode='same')

        # Scale the smoothed envelope
        max_value = np.max(smoothed_amplitude_envelope)
        if max_value > 0:
            scaled_envelope = scaling_factor * smoothed_amplitude_envelope / max_value

        # Apply a threshold
        scaled_envelope[scaled_envelope < threshold] = 0

        # Assuming extract_dominant_frequency is a defined function elsewhere
        dominant_frequency = extract_dominant_frequency(audio)

        return scaled_envelope, dominant_frequency

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def find_and_analyze_chirps(signal_envelope, amplitude_threshold):

    window_size, step_size, continuous_threshold = 100, 100, 100
    syllable_length = 700  # slightly lower than average length of 800
    zero_threshold = 100
    zero_tolerance = 5
    syllable_positions = []
    start = 0

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
    max_check_length = 300
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


def find_syllable_ends(signal_envelope, syllable_positions, zero_values_threshold=64, tolerance_percent=1):
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

            if distance_to_current < 2000:
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

        if end - start >= 200:
            paired_starts.append(start)
            paired_ends.append(end)

        # Move to the next valid start
        next_start_idx = start_idx + 1
        while next_start_idx < len(syllables_starts) and syllables_starts[next_start_idx] - start < 800:

            next_start_idx += 1

        start_idx = next_start_idx
        end_idx += 1

    return paired_starts, paired_ends, mismatch




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

    # Consider only the positive frequencies
    positive_frequencies = frequencies[:len(frequencies)//2]
    positive_magnitude = np.abs(fft_result)[:len(frequencies)//2]

    # Find the index of the maximum magnitude in the positive frequencies
    dominant_index = np.argmax(positive_magnitude)

    # Find the dominant frequency (positive only)
    dominant_frequency = int(positive_frequencies[dominant_index])

    return dominant_frequency

def evaluate_threshold(signal_envelope, threshold):
    syllables_starts = find_and_analyze_chirps(signal_envelope, amplitude_threshold=threshold)
    syllable_ends = find_syllable_ends(signal_envelope, syllables_starts)
    syllables_starts, syllable_ends, _ = match_starts_and_ends(syllables_starts, syllable_ends)

    chirps, _, _, _, single_syllable_groups = group_syllables_and_calculate_distances(syllables_starts, syllable_ends)

    return single_syllable_groups

def find_optimal_threshold(signal_envelope, start_threshold, end_threshold, step):
    optimal_threshold = start_threshold
    min_single_syllable_groups = float('inf')

    for threshold in np.arange(start_threshold, end_threshold, step):
        single_syllable_groups = evaluate_threshold(signal_envelope, threshold)
        if single_syllable_groups < min_single_syllable_groups:
            min_single_syllable_groups = single_syllable_groups
            optimal_threshold = threshold

    return optimal_threshold

#################################################################################################
#################################################################################################
#################################################################################################


print("The script is starting...")

sample_rate = 44100
mismatch = 0
default_threshold = 0.07
smoothing_window = 30
optimal_threshold = default_threshold
csv_filename = fr'D:\win{smoothing_window}th{default_threshold}.csv'
bugged_files = []

erase_modified_files()
files_paths = select_folder_experiment()

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["Experiment", "Subject", "File", "Total Syllable Starts", "Total Syllable Ends", "Total Chirps",
                     "Chirp Sizes Proportion", "Mean Intra-Group", "Std Intra-Group", "Mean Inter-Group",
                     "Std Inter-Group", "Mean syllable size", "Mean chirp size", "Dominant frequency", "optimal_threshold"])

    for file_path in files_paths:
        if file_path is None:
            continue
        print('Currently processing :', file_path)

        parts = file_path.split(os.sep)

        # Extract required details from the path
        experiment = parts[-3]  # Assuming the experiment name is 3 levels up from the file
        subject = parts[-2]  # Assuming the subject name is 2 levels up from the file

        signal_envelope, dominant_frequency = process_audio(file_path, smoothing_window=smoothing_window, threshold=default_threshold)
        signal_length = len(signal_envelope)

        syllables_starts = find_and_analyze_chirps(signal_envelope, amplitude_threshold=default_threshold)
        syllable_ends = find_syllable_ends(signal_envelope, syllables_starts)
        syllables_starts, syllable_ends, mismatch = match_starts_and_ends(syllables_starts, syllable_ends)
        mean_size_syllable = mean_syllable_size(syllables_starts, syllable_ends)

        chirps, intra_distances, inter_distances, group_size = group_syllables_and_calculate_distances(
            syllables_starts, syllable_ends)
        # if groups_one > 10:
        #     print("Optimizing threshold...")
        #     optimal_threshold = find_optimal_threshold(signal_envelope, start_threshold=default_threshold-0.04,
        #                                                end_threshold=0.08, step=0.01)
        #     syllables_starts = find_and_analyze_chirps(signal_envelope, amplitude_threshold=optimal_threshold)
        #     syllable_ends = find_syllable_ends(signal_envelope, syllables_starts)
        #     syllables_starts, syllable_ends, mismatch = match_starts_and_ends(syllables_starts, syllable_ends)
        #     chirps, intra_distances, inter_distances, group_size, _ = group_syllables_and_calculate_distances(
        #         syllables_starts, syllable_ends)
        mean_chirp_size = mean_group_size(group_size)
        chirps_proportion_result = chirps_proportion(chirps)

        intra_mean = np.mean(intra_distances) if intra_distances else 0
        intra_std = np.std(intra_distances) if intra_distances else 0

        # Calculate mean and standard deviation for intergroup distances
        inter_mean = np.mean(inter_distances) if inter_distances else 0
        inter_std = np.std(inter_distances) if inter_distances else 0

        writer.writerow([experiment, subject, os.path.basename(file_path), len(syllables_starts), len(syllable_ends),
                         len(chirps), chirps_proportion_result, round(intra_mean, 0), round(intra_std, 0),
                         round(inter_mean, 0), round(inter_std, 0), round(mean_size_syllable, 0),
                         round(mean_chirp_size, 0), dominant_frequency, optimal_threshold])

        create_modified_audio(file_path, syllables_starts, syllable_ends, chirps)


