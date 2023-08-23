# import soundfile as sf
import json
import gzip
# import matplotlib.pyplot as plt
# import librosa
from tkinter import filedialog
from tkinter import Tk
import numpy as np
import os
import shutil
import random
import h5py
def extract_subfolder_from_filename(filename):
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[1]
    return None


def get_file_info(folder_path):
    file_info = {}
    subfolders = set()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.gz'):
                subfolder = extract_subfolder_from_filename(file)
                if subfolder:
                    subfolders.add(subfolder)
                    file_path = os.path.join(root, file)
                    with gzip.open(file_path, 'rt') as f:
                        dict_data = json.load(f)
                        filename = dict_data["filename"]
                        key = f"{subfolder}/{filename}"
                        file_info[key] = file_path
    return file_info, subfolders


def get_train_subfolders(train_folder):
    subfolders = set()
    for root, dirs, files in os.walk(train_folder):
        for file in files:
            if file.endswith('.gz'):
                subfolder = extract_subfolder_from_filename(file)
                if subfolder:
                    subfolders.add(subfolder)
    return subfolders


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

# copy_missing_wav_files(treatment_mapping)

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


def convert_gzip_to_hdf5():
    # Create a tkinter root window and immediately hide it
    root = Tk()
    root.withdraw()

    # Ask the user to choose a folder
    folder_path = filedialog.askdirectory(title="Choose the folder containing gzip files")
    if not folder_path:
        print("No folder selected!")
        return

    output_folder = folder_path + '-h5py'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Walk through the chosen folder and its nested structure
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".gz"):
                gzip_path = os.path.join(dirpath, filename)

                # Extract the relative path
                relative_path = os.path.relpath(gzip_path, folder_path)

                # Determine the output path
                hdf5_path = os.path.join(output_folder, relative_path.replace('.gz', '.h5'))

                # Ensure the directory exists
                os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

                # Load JSON from gzip and save it to HDF5
                with gzip.open(gzip_path, 'rt') as gz_file:
                    data = json.load(gz_file)
                    with h5py.File(hdf5_path, 'w') as hf:
                        for key, value in data.items():
                            # Check if the value is a list or array-like
                            if isinstance(value, list) or isinstance(value, dict):
                                hf.create_dataset(key, data=value, compression="gzip")
                            else:
                                hf.create_dataset(key, data=value)

    print(f"Converted gzip files in {folder_path} to HDF5 files in {output_folder}.")

# Call the function
# convert_gzip_to_hdf5()
def plot_mfcc_from_h5():
    # Using tkinter to create a file dialog
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    file_path = filedialog.askopenfilename(title="Select an .h5 file",
                                           filetypes=(("h5 files", "*.h5"), ("all files", "*.*")))

    # Check if the user selected a file or canceled the dialog
    if not file_path:
        return

    # Open the h5 file
    with h5py.File(file_path, 'r') as f:
        # Assuming the MFCCs are stored in a dataset named 'mfcc' within the .h5 file
        # If the structure is different, adjust accordingly
        mfccs = f['mfcc'][:]

    # Check the shape to determine the number of segments
    num_segments = mfccs.shape[0]

    for segment_idx in range(num_segments):
        user_input = input(
            f"Press Enter to view MFCC for segment {segment_idx + 1}/{num_segments} or type 'exit' to stop: ")
        if user_input == 'exit':
            break
        plt.imshow(mfccs[segment_idx], cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar()
        plt.title(f"MFCC for Segment {segment_idx + 1}")
        plt.xlabel("Time")
        plt.ylabel("MFCC Coefficients")
        plt.tight_layout()
        plt.show()


plot_mfcc_from_h5()