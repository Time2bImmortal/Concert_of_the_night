import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import filedialog
# from main import DataLoader
from tensorflow import keras
import numpy as np
import os
import gzip
import json
import supporting_functions
import os
import pickle

class ModelEvaluator:
    def __init__(self, model_path, test_folder):
        self.model = keras.models.load_model(model_path)
        self.test_folder = test_folder
        with open('train_data_std_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        self.mean = scaler['mean']
        self.std = scaler['std']

    def evaluate(self, X, y):
        X = np.expand_dims(X, axis=-1)

        # Preprocess the data (normalization)
        X -= self.mean
        X /= self.std

        # Predict
        y_pred = self.model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred_classes)
        return cm

    def plot_confusion_matrix(self, confusion_mat, treatments_indices, title):
        # Get the treatment names sorted by their indices
        labels = [name for name, idx in sorted(treatments_indices.items(), key=lambda item: item[1])]

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.xticks(rotation=45)  # This can help if the labels are long and overlapping

        # Calculate confusion matrix results
        total_samples = confusion_mat.sum()
        correct_predictions = np.trace(confusion_mat)
        accuracy = correct_predictions / total_samples
        false_negatives = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
        false_positives = confusion_mat.sum(axis=0) - np.diag(confusion_mat)
        best_guess_class = np.argmax(confusion_mat, axis=1)

        test_directory = os.path.dirname(os.path.abspath(self.test_folder))

        # Create the 'DL_matrix_results' folder if it doesn't exist
        folder_path = os.path.join(test_directory, 'DL_MATRIX_RESULTS')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the confusion matrix plot in the 'DL_matrix_results' folder
        plot_file_path = os.path.join(folder_path, f'{title}.png')
        plt.savefig(plot_file_path)

        # Write confusion matrix results to a text file in the 'DL_matrix_results' folder
        txt_file_path = os.path.join(folder_path, f'{title}.txt')
        with open(txt_file_path, 'w') as file:
            file.write(f"Confusion Matrix Results for {title}\n")
            file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            file.write("\nFalse Negatives:\n")
            for idx, fn in enumerate(false_negatives):
                file.write(f"Class {labels[idx]}: {fn}\n")
            file.write("\nFalse Positives:\n")
            for idx, fp in enumerate(false_positives):
                file.write(f"Class {labels[idx]}: {fp}\n")
            file.write("\nBest Guess Classifications:\n")
            for idx, guess in enumerate(best_guess_class):
                file.write(f"Class {labels[idx]}: {labels[guess]}\n")

        plt.show()



