import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import filedialog
from main import DataLoader
from tensorflow import keras
import numpy as np
import os
import gzip
import json
import supporting_functions

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

    def plot_confusion_matrix(self, confusion_mat, treatments_indices):
        # Get the treatment names sorted by their indices
        labels = [name for name, idx in sorted(treatments_indices.items(), key=lambda item: item[1])]

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.xticks(rotation=45)  # This can help if the labels are long and overlapping
        plt.show()


if __name__ == "__main__":
    # Ask for the train and test folders
    root = tk.Tk()
    root.withdraw()
    treatments_indices = {'2lux': 0, '5lux': 1, 'LD': 2, 'LL': 3}
    print("Select the train folder")
    train_folder = filedialog.askdirectory()

    print("Select the test folder")
    test_folder = filedialog.askdirectory()

    # Delete common files
    supporting_functions.delete_common_files(train_folder, test_folder)

    # Load model
    model_path = 'saved_models/model_accuracy_0.92.h5'  # specify the correct model path
    evaluator = ModelEvaluator(model_path)

    # Assuming DataLoader is a class you have defined to load data
    # Load new data
    data_loader = DataLoader(test_folder, 100)  # Please make sure DataLoader class is defined somewhere

    # Evaluate
    confusion_mat = evaluator.evaluate(data_loader.X, data_loader.y)

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(confusion_mat, treatments_indices)
