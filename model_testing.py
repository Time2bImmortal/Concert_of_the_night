import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import filedialog
from main import DataLoader
from tensorflow import keras
import numpy as np
import os
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


def get_file_dict(folder_path):
    file_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                # Hash the content to create a unique identifier for the file
                content_hash = hashlib.sha256(f.read()).hexdigest()
            file_dict[content_hash] = file_path
    return file_dict


def delete_common_files(train_folder, test_folder):
    train_file_dict = get_file_dict(train_folder)
    test_file_dict = get_file_dict(test_folder)
    common_hashes = set(train_file_dict.keys()).intersection(set(test_file_dict.keys()))

    for common_hash in common_hashes:
        file_to_delete = test_file_dict[common_hash]
        try:
            os.remove(file_to_delete)
            print(f"Deleted: {file_to_delete}")
        except OSError as e:
            print(f"Error deleting {file_to_delete}: {e}")


if __name__ == "__main__":
    # Ask for the train and test folders
    root = tk.Tk()
    root.withdraw()

    print("Select the train folder")
    train_folder = filedialog.askdirectory()

    print("Select the test folder")
    test_folder = filedialog.askdirectory()

    # Delete common files
    delete_common_files(train_folder, test_folder)

    # Load model
    model_path = 'saved_models/model_accuracy_0.94.h5'  # specify the correct model path
    evaluator = ModelEvaluator(model_path)

    # Load new data (You can reuse DataLoader for this)
    data_loader = DataLoader(test_folder)

    # Evaluate
    confusion_mat = evaluator.evaluate(data_loader.X, data_loader.y)

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(confusion_mat)
