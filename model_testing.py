import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import filedialog
from main import DataLoader
from tensorflow import keras
import numpy as np
import os
import  gzip
import json
import hashlib
class ModelEvaluator:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def evaluate(self, X, y):
        X = np.expand_dims(X, axis=-1)

        # Preprocess the data (normalization)
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)

        # Predict
        y_pred = self.model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred_classes)
        return cm

    def plot_confusion_matrix(self, confusion_mat):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()


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
            if file.endswith('.gz'):
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


if __name__ == "__main__":
    # Ask for the train and test folders
    root = tk.Tk()
    root.withdraw()

    # print("Select the train folder")
    # train_folder = filedialog.askdirectory()

    print("Select the test folder")
    test_folder = filedialog.askdirectory()

    # Delete common files
    # delete_common_files(train_folder, test_folder)

    # Load model
    model_path = 'saved_models/model_accuracy_0.94.h5'  # specify the correct model path
    evaluator = ModelEvaluator(model_path)

    # Assuming DataLoader is a class you have defined to load data
    # Load new data
    data_loader = DataLoader(test_folder, 50)  # Please make sure DataLoader class is defined somewhere

    # Evaluate
    confusion_mat = evaluator.evaluate(data_loader.X, data_loader.y)

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(confusion_mat)
