from tkinter import filedialog
import tkinter as tk
from deep_learning_models import DataLoader, ModelTrainer

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()
    data_loader = DataLoader(folder_path, num_files_per_treatment=256)
    data_loader.split_data_files()
    model_trainer = ModelTrainer(data_loader)
    model_trainer.train_model(50)
    model_trainer.test_model()
