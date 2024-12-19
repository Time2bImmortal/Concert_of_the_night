import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
from scipy.signal import hilbert
import soundfile as sf
import csv
import os


class AudioAnalyzer:
    def __init__(self, sample_rate=44100, default_threshold=0.008, smoothing_window=20):
        self.sample_rate = sample_rate
        self.default_threshold = default_threshold
        self.smoothing_window = smoothing_window

    def process_audio(self, file_path):
        """Load and process audio file."""
        try:
            audio, _ = librosa.load(file_path, sr=self.sample_rate)
            dominant_frequency = self._extract_dominant_frequency(audio)

            # Normalize and scale audio
            audio = (audio / np.max(np.abs(audio))) * 0.9

            # Process envelope
            amplitude_envelope = np.abs(hilbert(audio))
            amplitude_envelope[amplitude_envelope < self.default_threshold * 5] = 0

            # Smooth envelope
            smoothed_envelope = np.convolve(
                amplitude_envelope,
                np.ones(self.smoothing_window) / self.smoothing_window,
                mode='same'
            )
            smoothed_envelope[smoothed_envelope < self.default_threshold * 3] = 0

            return smoothed_envelope, dominant_frequency
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None, None

    def _extract_dominant_frequency(self, signal):
        """Extract dominant frequency from signal."""
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(fft_result), 1 / self.sample_rate)

        positive_frequencies = frequencies[:len(frequencies) // 2]
        positive_magnitude = np.abs(fft_result)[:len(frequencies) // 2]

        dominant_index = np.argmax(positive_magnitude)
        return int(positive_frequencies[dominant_index])

    def find_syllables(self, signal_envelope):
        """Find syllable start positions."""
        window_size = 150
        step_size = 150
        continuous_threshold = 150
        syllable_length = 700
        zero_threshold = 100
        zero_tolerance = 1

        syllable_positions = []
        start = 0

        while start <= len(signal_envelope) - window_size:
            window = signal_envelope[start:start + window_size]
            if np.sum(window > self.default_threshold) >= continuous_threshold:
                refined_position = self._refine_position(
                    signal_envelope, start, zero_threshold, zero_tolerance
                )
                if refined_position != -1:
                    syllable_positions.append(refined_position)
                    start = refined_position + syllable_length
                    continue
            start += step_size

        return syllable_positions

    def _refine_position(self, signal, start_position, zero_threshold, zero_tolerance):
        """Refine syllable position."""
        max_check_length = 500
        end_check = max(0, start_position - max_check_length)
        segment = signal[end_check:start_position]

        if segment.size < zero_threshold:
            return -1

        segment_reversed = segment[::-1]
        is_zero = segment_reversed == 0
        window_sum = np.convolve(is_zero, np.ones(zero_threshold, dtype=int), 'valid')

        valid_positions = np.where(window_sum >= (zero_threshold - zero_tolerance))[0]
        if valid_positions.size > 0:
            return start_position - valid_positions[0] - 1

        return -1

    def find_syllable_ends(self, signal_envelope, syllable_positions):
        """Find syllable end positions."""
        ends = []
        for pos in syllable_positions:
            end_search_start = pos + 600
            end_search_end = min(pos + 2000, len(signal_envelope))

            if end_search_end - end_search_start >= 100:
                windowed_signal = np.lib.stride_tricks.sliding_window_view(
                    signal_envelope[end_search_start:end_search_end], 100
                )
                zero_counts = np.sum(windowed_signal == 0, axis=1)
                valid_windows = np.where(zero_counts >= 95)[0]

                if valid_windows.size > 0:
                    ends.append(end_search_start + valid_windows[0])

        return ends

    def match_syllables(self, starts, ends):
        """Match syllable starts with ends."""
        mismatch = 0

        # Clean up mismatched starts and ends
        while ends and starts and ends[0] < starts[0]:
            ends.pop(0)
            mismatch += 1
        while starts and ends and starts[-1] > ends[-1]:
            starts.pop()
            mismatch += 1

        paired_starts = []
        paired_ends = []
        start_idx = end_idx = 0

        while start_idx < len(starts) and end_idx < len(ends):
            start = starts[start_idx]

            while end_idx < len(ends) and ends[end_idx] < start:
                end_idx += 1

            if end_idx == len(ends) or (
                    start_idx + 1 < len(starts) and
                    starts[start_idx + 1] < ends[end_idx]
            ):
                start_idx += 1
                continue

            end = ends[end_idx]

            if end - start >= 300:
                paired_starts.append(start)
                paired_ends.append(end)

            next_start_idx = start_idx + 1
            while (next_start_idx < len(starts) and
                   starts[next_start_idx] - start < 600):
                next_start_idx += 1

            start_idx = next_start_idx
            end_idx += 1

        return paired_starts, paired_ends, mismatch


class ChirpAnalyzer:
    @staticmethod
    def analyze_groups(starts, ends, max_groups=3500):
        """Analyze syllable groups and calculate comprehensive statistics."""
        if len(starts) != len(ends):
            raise ValueError("Mismatched starts and ends")

        syllable_sizes = [end - start for start, end in zip(starts, ends)]
        mean_syllable_size, std_syllable_size = ChirpAnalyzer._mean_std(syllable_sizes)

        groups = []
        current_group = []
        intra_distances = []
        last_end = None

        for start, end in zip(starts, ends):
            if not current_group or start - current_group[-1][1] < 2000:
                if current_group:
                    intra_distances.append(start - current_group[-1][1])
                current_group.append((start, end))
            else:
                group_info = ChirpAnalyzer._process_group(
                    current_group, intra_distances, last_end, start
                )
                groups.append(group_info)
                last_end = end
                current_group = [(start, end)]
                intra_distances = []

        # Add last group if exists
        if current_group:
            group_info = ChirpAnalyzer._process_group(
                current_group, intra_distances, last_end, None
            )
            groups.append(group_info)

        return ChirpAnalyzer._calculate_statistics(groups, max_groups, mean_syllable_size, std_syllable_size)

    @staticmethod
    def _mean_std(data):
        """Calculate mean and standard deviation."""
        return (np.mean(data), np.std(data)) if data else (0, 0)

    @staticmethod
    def _process_group(group, distances, last_end, current_start):
        """Process individual group information."""
        num_syllables = len(group)
        sizes = [end - start for start, end in group]
        total_size = sum(sizes)
        mean_size = total_size / num_syllables if num_syllables else 0
        group_size = group[-1][1] - group[0][0]
        position = int((group[0][0] + group[-1][1]) / 2)
        positions = [start for start, _ in group]

        inter_distance = (
            current_start - last_end
            if last_end and current_start and 2000 < current_start - last_end < 14000
            else 0
        )

        mean_intra = (
            sum(distances) / len(distances)
            if num_syllables > 1 and distances
            else 0
        )

        return (
            num_syllables, group_size, mean_intra,
            position, mean_size, positions, inter_distance
        )

    '''@staticmethod
    def _calculate_statistics(groups, max_groups, mean_syllable_size, std_syllable_size):
        """Calculate comprehensive group statistics."""
        # Filter groups
        groups_3_4 = [g for g in groups if g[0] in [3, 4]][:max_groups]
        groups_3 = [g for g in groups_3_4 if g[0] == 3]
        groups_4 = [g for g in groups_3_4 if g[0] == 4]

        # Calculate basic statistics
        all_group_sizes = [g[1] for g in groups]
        all_intra_distances = [g[2] for g in groups if g[2] > 0]
        mean_group_size, std_group_size = ChirpAnalyzer._mean_std(all_group_sizes)
        mean_intra, std_intra = ChirpAnalyzer._mean_std(all_intra_distances)

        # Calculate statistics for size 3 groups
        sizes_3 = [g[1] for g in groups_3]
        intra_3 = [g[2] for g in groups_3]
        mean_size_3, std_size_3 = ChirpAnalyzer._mean_std(sizes_3)
        mean_intra_3, std_intra_3 = ChirpAnalyzer._mean_std(intra_3)

        # Calculate statistics for size 4 groups
        sizes_4 = [g[1] for g in groups_4]
        intra_4 = [g[2] for g in groups_4]
        mean_size_4, std_size_4 = ChirpAnalyzer._mean_std(sizes_4)
        mean_intra_4, std_intra_4 = ChirpAnalyzer._mean_std(intra_4)

        # Calculate combined 3/4 statistics
        syllable_sizes_3_4 = [g[4] for g in groups_3_4]
        intra_distances_3_4 = [g[2] for g in groups_3_4]
        mean_syllable_3_4, std_syllable_3_4 = ChirpAnalyzer._mean_std(syllable_sizes_3_4)
        mean_intra_3_4, std_intra_3_4 = ChirpAnalyzer._mean_std(intra_distances_3_4)

        # Calculate mean syllable sizes
        mean_syllable_3 = np.mean([g[4] for g in groups_3]) if groups_3 else 0
        mean_syllable_4 = np.mean([g[4] for g in groups_4]) if groups_4 else 0

        # Calculate inter-distances statistics
        all_inter = [g[6] for g in groups if g[6] != 0]
        inter_3_3 = [g2[6] for g1, g2 in zip(groups[:-1], groups[1:])
                     if g1[0] == 3 and g2[0] == 3 and g2[6] != 0]
        inter_4_4 = [g2[6] for g1, g2 in zip(groups[:-1], groups[1:])
                     if g1[0] == 4 and g2[0] == 4 and g2[6] != 0]
        inter_3_4 = [g2[6] for g1, g2 in zip(groups[:-1], groups[1:])
                     if g1[0] == 3 and g2[0] == 4 and g2[6] != 0]
        inter_4_3 = [g2[6] for g1, g2 in zip(groups[:-1], groups[1:])
                     if g1[0] == 4 and g2[0] == 3 and g2[6] != 0]

        # Calculate means and standard deviations for inter-distances
        mean_inter, std_inter = ChirpAnalyzer._mean_std(all_inter)
        mean_inter_3_3, std_inter_3_3 = ChirpAnalyzer._mean_std(inter_3_3)
        mean_inter_4_4, std_inter_4_4 = ChirpAnalyzer._mean_std(inter_4_4)
        mean_inter_3_4, std_inter_3_4 = ChirpAnalyzer._mean_std(inter_3_4)
        mean_inter_4_3, std_inter_4_3 = ChirpAnalyzer._mean_std(inter_4_3)

        return {
            "Total chirps": len(groups),
            "chirps_proportion": ChirpAnalyzer._calculate_proportion(groups),
            "mean_intra_group_distance": mean_intra,
            "std_intra_group_distance": std_intra,
            "mean_syllable_size": mean_syllable_size,
            "std_syllable_size": std_syllable_size,
            "mean_group_size": mean_group_size,
            "std_group_size": std_group_size,
            "total 3 chirps": len(groups_3),
            "total 4 chirps": len(groups_4),
            "mean_size_3": mean_size_3,
            "std_size_3": std_size_3,
            "mean_size_4": mean_size_4,
            "std_size_4": std_size_4,
            "mean_intra_distance_3": mean_intra_3,
            "std_intra_distance_3": std_intra_3,
            "mean_intra_distance_4": mean_intra_4,
            "std_intra_distance_4": std_intra_4,
            "total_filtered_groups": len(groups_3_4),
            "mean_syllable_size_3_and_4": mean_syllable_3_4,
            "std_syllable_size_3_and_4": std_syllable_3_4,
            "mean_intra_size_3_and_4": mean_intra_3_4,
            "std_intra_size_3_and_4": std_intra_3_4,
            "mean syllable size 3": mean_syllable_3,
            "mean syllable size 4": mean_syllable_4,
            "all_mean_inter": mean_inter,
            "all_std_inter": std_inter,
            "3_3_mean_inter": mean_inter_3_3,
            "3_3_std_inter": std_inter_3_3,
            "4_4_mean_inter": mean_inter_4_4,
            "4_4_std_inter": std_inter_4_4,
            "3_4_mean_inter": mean_inter_3_4,
            "3_4_std_inter": std_inter_3_4,
            "4_3_mean_inter": mean_inter_4_3,
            "4_3_std_inter": std_inter_4_3,
            "inter distances num": len(all_inter),
            "inter distances 3_3": len(inter_3_3),
            "inter distances 4_4": len(inter_4_4),
            "inter distances 3_4": len(inter_3_4),
            "inter distances 4_3": len(inter_4_3)
        }, groups, [g[5] for g in groups_3], [g[5] for g in groups_4]'''

    @staticmethod
    def _calculate_statistics(groups, max_groups, mean_syllable_size, std_syllable_size):
        """Calculate comprehensive group statistics."""
        # Filter groups
        groups_3_4 = [g for g in groups if g[0] in [3, 4]]
        groups_3_4_limited = groups_3_4[:max_groups]  # First max_groups for 3500 calculations

        groups_3 = [g for g in groups_3_4 if g[0] == 3]
        groups_4 = [g for g in groups_3_4 if g[0] == 4]
        groups_3_limited = [g for g in groups_3_4_limited if g[0] == 3]
        groups_4_limited = [g for g in groups_3_4_limited if g[0] == 4]

        # Calculate basic statistics
        all_group_sizes = [g[1] for g in groups]
        all_intra_distances = [g[2] for g in groups if g[2] > 0]
        mean_group_size, std_group_size = ChirpAnalyzer._mean_std(all_group_sizes)
        mean_intra, std_intra = ChirpAnalyzer._mean_std(all_intra_distances)

        # Calculate statistics for all 3/4 groups
        syllable_sizes_3_4 = [g[4] for g in groups_3_4]
        intra_distances_3_4 = [g[2] for g in groups_3_4]
        mean_syllable_3_4, std_syllable_3_4 = ChirpAnalyzer._mean_std(syllable_sizes_3_4)
        mean_intra_3_4, std_intra_3_4 = ChirpAnalyzer._mean_std(intra_distances_3_4)

        # Calculate statistics for limited 3500 3/4 groups
        syllable_sizes_3_4_limited = [g[4] for g in groups_3_4_limited]
        intra_distances_3_4_limited = [g[2] for g in groups_3_4_limited]
        mean_syllable_3_4_limited, std_syllable_3_4_limited = ChirpAnalyzer._mean_std(syllable_sizes_3_4_limited)
        mean_intra_3_4_limited, std_intra_3_4_limited = ChirpAnalyzer._mean_std(intra_distances_3_4_limited)

        # Calculate statistics for size 3 groups (limited)
        sizes_3 = [g[4] for g in groups_3_limited]
        intra_3 = [g[2] for g in groups_3_limited]
        mean_size_3, std_size_3 = ChirpAnalyzer._mean_std(sizes_3)
        mean_intra_3, std_intra_3 = ChirpAnalyzer._mean_std(intra_3)

        # Calculate statistics for size 4 groups (limited)
        sizes_4 = [g[4] for g in groups_4_limited]
        intra_4 = [g[2] for g in groups_4_limited]
        mean_size_4, std_size_4 = ChirpAnalyzer._mean_std(sizes_4)
        mean_intra_4, std_intra_4 = ChirpAnalyzer._mean_std(intra_4)

        return {
            "Total chirps": len(groups),
            "mean_syllable_size": mean_syllable_size,
            "std_syllable_size": std_syllable_size,
            "total_filtered_groups": len(groups_3_4),
            "mean_syllable_size_3_and_4": mean_syllable_3_4,
            "std_syllable_size_3_and_4": std_syllable_3_4,
            "mean_intra_size_3_and_4": mean_intra_3_4,
            "std_intra_size_3_and_4": std_intra_3_4,
            "limited_mean_syllable_3_4": mean_syllable_3_4_limited,
            "limited_std_syllable_3_4": std_syllable_3_4_limited,
            "limited_mean_intra_3_4": mean_intra_3_4_limited,
            "limited_std_intra_3_4": std_intra_3_4_limited,
            "mean_size_3": mean_size_3,
            "std_size_3": std_size_3,
            "mean_intra_distance_3": mean_intra_3,
            "std_intra_distance_3": std_intra_3,
            "mean_size_4": mean_size_4,
            "std_size_4": std_size_4,
            "mean_intra_distance_4": mean_intra_4,
            "std_intra_distance_4": std_intra_4
        }, groups, [g[5] for g in groups_3], [g[5] for g in groups_4]
    @staticmethod
    def _calculate_proportion(groups):
        """Calculate proportion of different group sizes."""
        counts = {}
        for num_syllables, *_ in groups:
            counts[num_syllables] = counts.get(num_syllables, 0) + 1
        max_size = max(counts.keys(), default=0)
        return [counts.get(i, 0) for i in range(1, max_size + 1)]


def create_modified_audio(file_path, syllables_starts, syllables_ends, group_info):
    """Create modified audio file with markers and labels."""
    # Load the original audio file
    audio, _ = librosa.load(file_path, sr=None)

    # Create a copy of the audio data
    modified_audio = np.copy(audio)

    # Set markers at syllable positions
    for pos in syllables_starts:
        if pos < len(modified_audio):
            modified_audio[int(pos)] = 1
    for pos in syllables_ends:
        if pos < len(modified_audio):
            modified_audio[int(pos)] = -1

    # Define output paths
    output_audio_path = file_path.replace('.wav', '_modified.wav')
    output_label_path = file_path.replace('.wav', '_labels.txt')

    # Save modified audio
    sf.write(output_audio_path, modified_audio, 44100)

    # Create and save labels
    with open(output_label_path, 'w') as label_file:
        for num_syllables, _, _, middle_position, _, _, _ in group_info:
            time_position = middle_position / 44100
            label_file.write(f"{time_position}\t{time_position}\tGroup Size: {num_syllables}\n")

    print(f"Modified audio saved as: {output_audio_path}")
    print(f"Labels saved as: {output_label_path}")

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

def main():
    """Main execution function."""
    print("Starting audio analysis...")
    sample_rate = 44100
    default_threshold = 0.008
    smoothing_window = 20

    erase_modified_files()
    files_paths = select_folder_experiment()
    analyzer = AudioAnalyzer()
    csv_filename = '/home/yossef-aidan/Desktop/3500_correction.csv'


    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        '''writer.writerow(
            ["Experiment", "Subject", "File", "Total Syllable Starts", "Total Syllable Ends", "Total Chirps",
             "Chirps Proportion", " All Mean Intra-Group", " All Std Intra-Group", " All Mean syllable size",

             "All Std syllable size", " All Mean chirp size", " All Std chirp size",
             "Dominant frequency", "optimal_threshold",

             "3 chirps number", "mean 3 chirps size", "std size 3 chirps", "mean intra 3", "std intra 3",
             "4 chirps number", "mean 4 chirps size", "std size 4 chirps", "mean intra 4", "std intra 4",
             "total 3/4 chirps", "mean syllables size 3/4", "std syllables 3/4", "mean intra size 3/4",
             "std intra size 3/4",
             "syllable size 3", "syllable size 4",

             "inter_all_mean", "inter_all_std", "inter_3_3_mean", "inter_3_3_std", "inter_4_4_mean",
             "inter_4_4_std", "inter_3_4_mean", "inter_3_4_std", "inter_4_3_mean", "inter_4_3_std",
             "all inter num", "3_3 num", "4_4 num", "3_4 num", "4_3 num"])'''
        writer.writerow(
            ["Experiment", "Subject", "File", "Total Syllables", " All Mean syllable size", "All Std syllable size",

             "all 3/4 syllables", "mean all syllables size 3/4", "std all syllables 3/4", "mean all intra size 3/4",
             "std all intra size 3/4",
             "3500 mean syllables s3/4", "3500 std syllables s3/4", "3500 intra mean s3/4", "3500 intra std s3/4",
             "3500 mean s3", "3500 std s3", "3500 intra mean s3", "3500 intra std s3",
             "3500 mean s4", "3500 std s4", "3500 intra mean s4", "3500 intra std s4"])

        for file_path in files_paths:
            if not file_path:
                continue

            print(f'Processing: {file_path}')

            # Extract path components
            parts = file_path.split(os.sep)
            experiment = parts[-3]
            subject = parts[-2]

            # Process audio
            envelope, freq = analyzer.process_audio(file_path)
            if envelope is None:
                continue

            # Find syllables
            starts = analyzer.find_syllables(envelope)
            ends = analyzer.find_syllable_ends(envelope, starts)
            starts, ends, mismatch = analyzer.match_syllables(starts, ends)

            # Analyze groups
            results, groups, pos_3, pos_4 = ChirpAnalyzer.analyze_groups(starts, ends)

            # Round numeric values
            rounded_results = {k: round((v / 44100) * 1000, 4) if isinstance(v, (int, float)) else v
                               for k, v in results.items()}

            # Write row data
            '''writer.writerow([
                experiment, subject, os.path.basename(file_path),
                len(starts), len(ends), rounded_results["Total chirps"],
                rounded_results["chirps_proportion"],
                rounded_results["mean_intra_group_distance"],
                rounded_results["std_intra_group_distance"],
                rounded_results["mean_syllable_size"],
                rounded_results["std_syllable_size"],
                rounded_results["mean_group_size"],
                rounded_results["std_group_size"],
                freq, analyzer.default_threshold,
                rounded_results["total 3 chirps"],
                rounded_results["mean_size_3"],
                rounded_results["std_size_3"],
                rounded_results["mean_intra_distance_3"],
                rounded_results["std_intra_distance_3"],
                rounded_results["total 4 chirps"],
                rounded_results["mean_size_4"],
                rounded_results["std_size_4"],
                rounded_results["mean_intra_distance_4"],
                rounded_results["std_intra_distance_4"],
                rounded_results["total_filtered_groups"],
                rounded_results["mean_syllable_size_3_and_4"],
                rounded_results["std_syllable_size_3_and_4"],
                rounded_results["mean_intra_size_3_and_4"],
                rounded_results["std_intra_size_3_and_4"],
                rounded_results["mean syllable size 3"],
                rounded_results["mean syllable size 4"],
                rounded_results["all_mean_inter"],
                rounded_results["all_std_inter"],
                rounded_results["3_3_mean_inter"],
                rounded_results["3_3_std_inter"],
                rounded_results["4_4_mean_inter"],
                rounded_results["4_4_std_inter"],
                rounded_results["3_4_mean_inter"],
                rounded_results["3_4_std_inter"],
                rounded_results["4_3_mean_inter"],
                rounded_results["4_3_std_inter"],
                rounded_results["inter distances num"],
                rounded_results["inter distances 3_3"],
                rounded_results["inter distances 4_4"],
                rounded_results["inter distances 3_4"],
                rounded_results["inter distances 4_3"]
            ])'''
            writer.writerow([
                experiment,  # "Experiment"
                subject,  # "Subject"
                os.path.basename(file_path),  # "File"
                len(starts),  # "Total Syllables"
                rounded_results["mean_syllable_size"],  # "All Mean syllable size"
                rounded_results["std_syllable_size"],  # "All Std syllable size"

                rounded_results["total_filtered_groups"],  # "all 3/4 syllables"
                rounded_results["mean_syllable_size_3_and_4"],  # "mean all syllables size 3/4"
                rounded_results["std_syllable_size_3_and_4"],  # "std all syllables 3/4"
                rounded_results["mean_intra_size_3_and_4"],  # "mean all intra size 3/4"
                rounded_results["std_intra_size_3_and_4"],  # "std all intra size 3/4"

                # The following should be calculated for first 3500 groups only
                rounded_results["limited_mean_syllable_3_4"],  # for "3500 mean syllables s3/4"
                rounded_results["limited_std_syllable_3_4"],  # for "3500 std syllables s3/4"
                rounded_results["limited_mean_intra_3_4"],  # for "3500 intra mean s3/4"
                rounded_results["limited_std_intra_3_4"],  # "3500 intra std s3/4"

                rounded_results["mean_size_3"],  # "3500 mean s3"
                rounded_results["std_size_3"],  # "3500 std s3"
                rounded_results["mean_intra_distance_3"],  # "3500 intra mean s3"
                rounded_results["std_intra_distance_3"],  # "3500 intra std s3"

                rounded_results["mean_size_4"],  # "3500 mean s4"
                rounded_results["std_size_4"],  # "3500 std s4"
                rounded_results["mean_intra_distance_4"],  # "3500 intra mean s4"
                rounded_results["std_intra_distance_4"]  # "3500 intra std s4"
            ])

            # Create modified audio file
            #create_modified_audio(file_path, starts, ends, groups)

if __name__ == "__main__":
    main()