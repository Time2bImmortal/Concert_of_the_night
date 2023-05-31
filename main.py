import tkinter as tk
from tkinter import filedialog
import audioread
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import wavfile


def choose_folder():
    root = tk.Tk()
    # Hide the main window
    root.withdraw()

    folder_path = filedialog.askdirectory()
    return folder_path


def prepare_datasets(test_size, validation_size):
    pass


if __name__ == "__main__":


    #  create, train, validations sets
    # build the cnn net
    # train the cnn
    # Evaluate the CNN on the test set
    # Make predictions on a sample