import tkinter as tk
from tkinter import filedialog
from Features_extraction import *
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
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size)

    X_train, X_validation, X_test = X_train[..., np.newaxis], X_validation[..., np.newaxis], X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test
if __name__ == "__main__":

    #  create, train, validations sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build the cnn net
    model = build_model(input_shape)
    # train the cnn
    # Evaluate the CNN on the test set
    # Make predictions on a sample