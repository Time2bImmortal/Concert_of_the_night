import tkinter as tk
from tkinter import filedialog
import librosa
import librosa.display
import scipy as sp
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
def choose_folder():
    root = tk.Tk()
    # Hide the main window
    root.withdraw()

    folder_path = filedialog.askdirectory()
    return folder_path


if __name__ == "__main__":
    chosen_folder = choose_folder()
    print(f'Chosen folder: {chosen_folder}')
