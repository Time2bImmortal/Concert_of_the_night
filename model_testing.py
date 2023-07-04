import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import filedialog
from main import DataLoader
from tensorflow import keras
import numpy as np
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


if __name__ == "__main__":
    # Load model
    model_path = 'saved_models/model_accuracy_0.94.h5'  # specify the correct model path
    evaluator = ModelEvaluator(model_path)

    # Load new data (You can reuse DataLoader for this)
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    data_loader = DataLoader(folder_path)

    # Evaluate
    confusion_mat = evaluator.evaluate(data_loader.X, data_loader.y)

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(confusion_mat)
