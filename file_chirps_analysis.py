import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
from scipy.signal import hilbert
import soundfile as sf
import csv
import os

""" This code is designed to detect syllables and chirps in audio files, capturing their start and end points for "
 "documentation in an Excel file. It's versatile, allowing adjustments to fit different scenarios, and is 
 straightforward yet highly accurate. The code also creates visual copies of the audio for easy verification of results,
  making it suitable for machine learning projects and research."""


#  Supportive utility functions
def print_syllable_details(func):
    def wrapper(*args, **kwargs):
        syllable_position, similarity = func(*args, **kwargs)
        time_position = syllable_position / 44100  # Assuming a sample rate of 44100 Hz

        print(f"Syllable Position: {syllable_position} (Sample Number)")
        print(f"Time Position: {time_position:.2f} seconds")
        print(f"Similarity: {similarity}")

        return syllable_position, similarity
    return wrapper

def round_numeric_values(dictionary):
    rounded_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, (int, float)):
            rounded_dict[key] = round(value,0)
        else:
            rounded_dict[key] = value
    return rounded_dict


#  Standard functions for selecting files
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


def process_audio(file_path, smoothing_window=20, scaling_factor=0.9, threshold=0.08):
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=44100)
        dominant_frequency = extract_dominant_frequency(audio)

        # Normalize the audio and scale it
        audio = (audio / np.max(np.abs(audio))) * scaling_factor

        amplitude_envelope = np.abs(hilbert(audio))

        # Apply a threshold
        amplitude_envelope[amplitude_envelope < threshold*5] = 0


        # Smooth the amplitude envelope
        smoothed_amplitude_envelope = np.convolve(amplitude_envelope, np.ones(smoothing_window) / smoothing_window,
                                                  mode='same')
        smoothed_amplitude_envelope[smoothed_amplitude_envelope < threshold*3] = 0

        return smoothed_amplitude_envelope, dominant_frequency

    except Exception as e:
        print(f"Error processing audio: {e}")

def find_and_analyze_chirps(signal_envelope, amplitude_threshold):

    window_size, step_size, continuous_threshold = 150, 150, 150
    syllable_length = 700  # slightly lower than average length of 800
    zero_threshold = 100
    zero_tolerance = 1
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
    max_check_length = 500
    end_check = max(0, start_position - max_check_length)
    segment = signal[end_check:start_position]

    if segment.size < zero_threshold:
        return -1  # Segment is too short

    # Reverse the segment for backward search
    segment_reversed = segment[::-1]

    # Create a boolean array where zeros are 1 and non-zeros are 0
    is_zero = segment_reversed == 0

    # Apply a sliding window to sum over 'zero_threshold' consecutive elements
    window_sum = np.convolve(is_zero, np.ones(zero_threshold, dtype=int), 'valid')

    # Find the last position where the sum equals 'zero_threshold' (or within tolerance)
    valid_positions = np.where(window_sum >= (zero_threshold - zero_tolerance))[0]
    if valid_positions.size > 0:
        # Calculate the position in the original (non-reversed) segment
        last_valid_position = valid_positions[0]
        # Adjust to get the position at the start of the 100-zero segment
        return start_position - last_valid_position - 1

    return -1  # If no such position is found


def find_syllable_ends(signal_envelope, syllable_positions, zero_values_threshold=100, tolerance_percent=5):
    syllable_end_positions = []
    signal_array = np.array(signal_envelope)

    for position in syllable_positions:
        end_search_start = position + int(600)
        end_search_end = min(position + int(2000), len(signal_array))

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


def mean_and_standard_deviation(data):
    if len(data) == 0:  # Handle empty data case
        return 0, 0
    data_array = np.array(data)
    mean_value = np.mean(data_array)
    std_dev_value = np.std(data_array)
    return mean_value, std_dev_value


def group_and_analyze_syllables(syllable_starts, syllable_ends, max_groups_3_and_4=3500):
    if len(syllable_starts) != len(syllable_ends):
        raise ValueError("The lengths of syllable_starts and syllable_ends lists must be the same.")

    # Calculate syllable sizes
    syllable_sizes = [end - start for start, end in zip(syllable_starts, syllable_ends)]
    mean_syllable_size, std_syllable_size = mean_and_standard_deviation(syllable_sizes)
    groups_info = []  # To store (num_syllables, group_size, mean_intra_distances, position, mean_syllable_size)
    current_group = []
    intra_group_distances = []
    last_end = None

    for start, end in zip(syllable_starts, syllable_ends):
        if not current_group or start - current_group[-1][1] < 2000:
            if current_group:  # Calculate intra-group distance if not the first syllable
                intra_group_distances.append(start - current_group[-1][1])
            current_group.append((start, end))
        else:
            # Calculate and append group information
            num_syllables = len(current_group)
            group_syllable_sizes = [end - start for start, end in current_group]
            total_syllable_size = sum(group_syllable_sizes)
            mean_syllable_size = total_syllable_size / num_syllables if num_syllables else 0
            group_size = current_group[-1][1] - current_group[0][0]
            position = int((current_group[0][0] + current_group[-1][1]) / 2)  # Midpoint position
            syllable_positions = [start for start, _ in current_group]

            if last_end is not None:
                inter_distance = start - last_end if 2000 < start - last_end < 14000 else 0
            else:
                inter_distance = 0

            # Calculate mean intra-group distance if there are distances to calculate
            if num_syllables > 1 and intra_group_distances:
                mean_intra_distance = sum(intra_group_distances) / len(intra_group_distances)
            else:
                mean_intra_distance = 0  # Default to 0 for groups with a single syllable or no distances

            groups_info.append(
                (num_syllables, group_size, mean_intra_distance, position, mean_syllable_size, syllable_positions, inter_distance)
            )

            # Reset for next group
            last_end = end
            current_group = [(start, end)]
            intra_group_distances = []

    # Extract information for analysis
    all_group_sizes = [info[1] for info in groups_info]
    all_intra_distances = [info[2] for info in groups_info if info[2] > 0]

    combined_groups_3_and_4 = [info for info in groups_info if info[0] in [3, 4]]
    limited_combined_groups = combined_groups_3_and_4[:max_groups_3_and_4]
    groups_3 = [info for info in limited_combined_groups if info[0] == 3]
    groups_4 = [info for info in limited_combined_groups if info[0] == 4]

    syllable_sizes_3_and_4 = [info[4] for info in limited_combined_groups]
    intra_distances_3_and_4 = [info[2] for info in limited_combined_groups]
    syllable_size_3 = [info[4] for info in groups_3]
    syllable_size_4 = [info[4] for info in groups_4]

    mean_group_size, std_group_size = mean_and_standard_deviation(all_group_sizes)
    mean_intra, std_intra = mean_and_standard_deviation(all_intra_distances)
    mean_size_3, std_size_3 = mean_and_standard_deviation([info[1] for info in groups_3])
    mean_size_4, std_size_4 = mean_and_standard_deviation([info[1] for info in groups_4])
    mean_intra_3, std_intra_3 = mean_and_standard_deviation([info[2] for info in groups_3])
    mean_intra_4, std_intra_4 = mean_and_standard_deviation([info[2] for info in groups_4])
    mean_syllable_size_3_and_4, std_syllable_size_3_and_4 = mean_and_standard_deviation(syllable_sizes_3_and_4)
    mean_intra_size_3_and_4, std_intra_size_3_and_4 = mean_and_standard_deviation(intra_distances_3_and_4)
    mean_syllable_size_3, _ = mean_and_standard_deviation(syllable_size_3)
    mean_syllable_size_4, _ = mean_and_standard_deviation(syllable_size_4)

    chirps_proportion_result = chirps_proportion(groups_info)
    positions_3 = [info[5] for info in groups_3]
    positions_4 = [info[5] for info in groups_4]

    all_inter_distances = [info[6] for info in groups_info if info[6] != 0]
    all_mean_inter_distances, std_all_inter_distances = mean_and_standard_deviation(all_inter_distances)
    inter_distances_prev_4_current_4 = [info[6] for info, prev_info in zip(groups_info[1:], groups_info) if
                                          prev_info[0] == 4 and info[0] == 4 and info[6] != 0]
    inter_distances_prev_4_current_4_mean, inter_distances_prev_4_current_4_std = mean_and_standard_deviation(inter_distances_prev_4_current_4)
    inter_distances_prev_3_current_3 = [info[6] for info, prev_info in zip(groups_info[1:], groups_info) if
                                          prev_info[0] == 3 and info[0] == 3 and info[6] != 0]
    inter_distances_prev_3_current_3_mean, inter_distances_prev_3_current_3_std = mean_and_standard_deviation(inter_distances_prev_3_current_3)
    inter_distances_prev_3_current_4 = [info[6] for info, prev_info in zip(groups_info[1:], groups_info) if
                                        prev_info[0] == 3 and info[0] == 4 and info[6] != 0]
    inter_distances_prev_3_current_4_mean, inter_distances_prev_3_current_4_std = mean_and_standard_deviation(inter_distances_prev_3_current_4)
    inter_distances_prev_4_current_3 = [info[6] for info, prev_info in zip(groups_info[1:], groups_info) if
                                        prev_info[0] == 4 and info[0] == 3 and info[6] != 0]
    inter_distances_prev_4_current_3_mean, inter_distances_prev_4_current_3_std = mean_and_standard_deviation(inter_distances_prev_4_current_3)

    return {
        "mean_syllable_size": mean_syllable_size,
        "std_syllable_size": std_syllable_size,
        "mean_group_size": mean_group_size,
        "std_group_size": std_group_size,
        "chirps_proportion": chirps_proportion_result,
        "Total chirps": len(groups_info),
        "mean_intra_group_distance": mean_intra,
        "std_intra_group_distance": std_intra,
        "total 3 chirps": len(groups_3),
        "total 4 chirps": len(groups_4),
        "mean_intra_distance_3": mean_intra_3,
        "std_intra_distance_3": std_intra_3,
        "mean_intra_distance_4": mean_intra_4,
        "std_intra_distance_4": std_intra_4,
        "total_filtered_groups": len(limited_combined_groups),
        "mean_size_3": mean_size_3,
        "std_size_3": std_size_3,
        "mean_size_4": mean_size_4,
        "std_size_4": std_size_4,
        "mean syllable size 3": mean_syllable_size_3,
        "mean syllable size 4": mean_syllable_size_4,
        "mean_syllable_size_3_and_4": mean_syllable_size_3_and_4,
        "std_syllable_size_3_and_4": std_syllable_size_3_and_4,
        "mean_intra_size_3_and_4": mean_intra_size_3_and_4,
        "std_intra_size_3_and_4": std_intra_size_3_and_4,
        "all_mean_inter": all_mean_inter_distances,
        "all_std_inter": std_all_inter_distances,
        "3_3_mean_inter": inter_distances_prev_3_current_3_mean,
        "3_3_std_inter": inter_distances_prev_3_current_3_std,
        "4_4_mean_inter": inter_distances_prev_4_current_4_mean,
        "4_4_std_inter": inter_distances_prev_4_current_4_std,
        "3_4_mean_inter": inter_distances_prev_3_current_4_mean,
        "3_4_std_inter": inter_distances_prev_3_current_4_std,
        "4_3_mean_inter": inter_distances_prev_4_current_3_mean,
        "4_3_std_inter": inter_distances_prev_4_current_3_std,
        "inter distances num": len(all_inter_distances),
        "inter distances 3_3": len(inter_distances_prev_3_current_3),
        "inter distances 4_4": len(inter_distances_prev_4_current_4),
        "inter distances 3_4": len(inter_distances_prev_3_current_4),
        "inter distances 4_3": len(inter_distances_prev_4_current_3)


    }, groups_info, positions_3, positions_4


def create_modified_audio(file_path, syllables_positions, syllables_ends, group_info):
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
        for num_syllables, _, _, middle_position, _, _,_ in group_info:
            # Convert the middle position to time in seconds
            time_position = middle_position / 44100
            label_file.write(f"{time_position}\t{time_position}\tGroup Size: {num_syllables}\n")

    print(f"Modified audio file saved as: {output_audio_path}")
    print(f"Label track file saved as: {output_label_path}")


def chirps_proportion(groups_info):
    # Initialize a dictionary to count occurrences of each group size
    group_sizes_counts = {}

    # Iterate over each group's info in the list
    for num_syllables, _, _, _,_,_,_ in groups_info:
        # Increment the count for this group size, initializing if not present
        if num_syllables in group_sizes_counts:
            group_sizes_counts[num_syllables] += 1
        else:
            group_sizes_counts[num_syllables] = 1

    # Determine the maximum group size dynamically for the range
    max_group_size = max(group_sizes_counts.keys(), default=0)

    # Create a list with the count of groups for each size from 1 to max_group_size
    counts_list = [group_sizes_counts.get(i, 0) for i in range(1, max_group_size + 1)]

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

        if end - start >= 300:
            paired_starts.append(start)
            paired_ends.append(end)

        # Move to the next valid start
        next_start_idx = start_idx + 1
        while next_start_idx < len(syllables_starts) and syllables_starts[next_start_idx] - start < 600:

            next_start_idx += 1

        start_idx = next_start_idx
        end_idx += 1

    return paired_starts, paired_ends, mismatch


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


# Application-specific functions
def calculate_frequency_statistics(positions, signal):
    if not positions:
        return 0, 0  # Return 0 mean and standard deviation if no positions are provided
    frequencies = []  # List to store frequencies
    for pos in positions:
        # Calculate frequency from position data (you need to implement this)
        frequency = calculate_frequency_from_position(pos, signal)
        frequencies.append(frequency)
    mean_frequency, std_frequency = mean_and_standard_deviation(frequencies) # Calculate mean frequency

    return mean_frequency, std_frequency


def calculate_frequency_from_position(position, signal):
    window_size = 1000  # Window size around the position

    # Ensure the window doesn't extend beyond the signal boundaries
    start_index = position
    end_index = position+window_size

    # Extract the windowed segment of the signal
    windowed_signal = signal[start_index:end_index]

    # Perform FFT on the windowed segment
    fft_result = np.fft.fft(windowed_signal)

    # Frequency axis
    freqs = np.fft.fftfreq(len(windowed_signal), 1 / 44100)

    # Find the peak frequency
    peak_index = np.argmax(np.abs(fft_result))
    peak_frequency = freqs[peak_index]

    return peak_frequency


#################################################################################################
#################################################################################################
#################################################################################################


print("The script is starting...")

sample_rate = 44100
default_threshold = 0.008
smoothing_window = 20
csv_filename = r'D:\Final results with inter distances.csv'  # results path

erase_modified_files()
files_paths = select_folder_experiment()

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header
    writer.writerow(["Experiment", "Subject", "File", "Total Syllable Starts", "Total Syllable Ends", "Total Chirps",
                     "Chirps Proportion", " All Mean Intra-Group", " All Std Intra-Group", " All Mean syllable size",

                     "All Std syllable size", " All Mean chirp size", " All Std chirp size",
                     "Dominant frequency", "optimal_threshold",

                     "3 chirps number","mean 3 chirps size", "std size 3 chirps","mean intra 3", "std intra 3",
                     "4 chirps number","mean 4 chirps size", "std size 4 chirps","mean intra 4", "std intra 4",
                     "total 3/4 chirps", "mean syllables size 3/4", "std syllables 3/4","mean intra size 3/4", "std intra size 3/4",
                     "syllable size 3", "syllable size 4",

                     "inter_all_mean", "inter_all_std", "inter_3_3_mean", "inter_3_3_std", "inter_4_4_mean",
                     "inter_4_4_std", "inter_3_4_mean", "inter_3_4_std", "inter_4_3_mean", "inter_4_3_std",
                     "all inter num", "3_3 num", "4_4 num", "3_4 num", "4_3 num"])

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

        results_dict, group_info, positions_3, positions_4 = group_and_analyze_syllables(syllables_starts,syllable_ends,3500)

        # Write the data to the CSV file

        rounded_results_dict = round_numeric_values(results_dict)

        writer.writerow([experiment,subject,os.path.basename(file_path),len(syllables_starts),len(syllable_ends),
                        rounded_results_dict["Total chirps"],rounded_results_dict["chirps_proportion"],
                        rounded_results_dict["mean_intra_group_distance"], rounded_results_dict["std_intra_group_distance"],
                        rounded_results_dict["mean_syllable_size"], rounded_results_dict["std_syllable_size"],
                        rounded_results_dict["mean_group_size"], rounded_results_dict["std_group_size"], dominant_frequency,
                        default_threshold, rounded_results_dict["total 3 chirps"], rounded_results_dict["mean_size_3"],
                        rounded_results_dict["std_size_3"], rounded_results_dict["mean_intra_distance_3"],
                         rounded_results_dict["std_intra_distance_3"], rounded_results_dict["total 4 chirps"],
                         rounded_results_dict["mean_size_4"], rounded_results_dict["std_size_4"],
                        rounded_results_dict["mean_intra_distance_4"], rounded_results_dict["std_intra_distance_4"],
                        rounded_results_dict["total_filtered_groups"], rounded_results_dict["mean_syllable_size_3_and_4"],
                         rounded_results_dict["std_syllable_size_3_and_4"], rounded_results_dict["mean_intra_size_3_and_4"],
                         rounded_results_dict["std_intra_size_3_and_4"], rounded_results_dict["mean syllable size 3"],
                         rounded_results_dict["mean syllable size 4"], rounded_results_dict["all_mean_inter"],
                         rounded_results_dict["all_std_inter"], rounded_results_dict["3_3_mean_inter"],
                         rounded_results_dict["3_3_std_inter"], rounded_results_dict["4_4_mean_inter"],
                         rounded_results_dict["4_4_std_inter"], rounded_results_dict["3_4_mean_inter"],
                         rounded_results_dict["3_4_std_inter"], rounded_results_dict["4_3_mean_inter"],
                         rounded_results_dict["4_3_std_inter"], rounded_results_dict["inter distances num"],
                         rounded_results_dict["inter distances 3_3"], rounded_results_dict["inter distances 4_4"],
                         rounded_results_dict["inter distances 3_4"], rounded_results_dict["inter distances 4_3"]])

        create_modified_audio(file_path, syllables_starts, syllable_ends, group_info)


