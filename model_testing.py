import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import torch
import supporting_functions
from pytorch_dl import CustomDataset, CustomDataLoader, DataLoader, MFCC_CNN, compute_mean_std
from sklearn.preprocessing import LabelEncoder


class ModelEvaluator:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MFCC_CNN().to(self.device)

        # Load the state dictionary
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set model to evaluation mode

    def evaluate(self, dataloader):
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, y_pred_classes = torch.max(outputs, 1)
                all_preds.extend(y_pred_classes.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        cm = confusion_matrix(all_targets, all_preds)
        return cm

    def analysis(self, cm, treatments_indices):
        # Calculating the accuracy for each class
        num_correct = np.diagonal(cm)
        total_per_class = np.sum(cm, axis=1)
        accuracy_per_class = num_correct / total_per_class

        general_accuracy = np.sum(num_correct) / np.sum(total_per_class)

        # Prepare class-wise accuracy data
        class_accuracies = [(label, acc) for label, acc in zip(treatments_indices.values(), accuracy_per_class)]
        class_accuracies.sort(key=lambda x: x[1], reverse=True)

        best_guess = class_accuracies[0]
        worst_guess = class_accuracies[-1]

        return best_guess, worst_guess, class_accuracies, general_accuracy

    def save_and_show_results(self, cm, treatments_indices, test_folder, show=False, save=True):
        best_guess, worst_guess, class_accuracies, general_accuracy = self.analysis(cm, treatments_indices)

        if save:
            # Creating a results directory
            folder_name = os.path.basename(test_folder)
            parent_directory = os.path.dirname(test_folder)
            result_dir = os.path.join(parent_directory, f"result_{folder_name}")
            os.makedirs(result_dir, exist_ok=True)

        # Save and/or show the confusion matrix image
        plt.figure()
        # Custom ordering function for the keys
        order = {'2': 0, '0': 1, '1': 2, '3': 3}
        sorted_keys = sorted(treatments_indices.keys(), key=lambda x: order[x])

        labels = [treatments_indices[key] for key in sorted_keys]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)

        if save:
            plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))

        if show:
            plt.show()

        plt.close()

        if save:
            # Save analysis to a text file
            with open(os.path.join(result_dir, 'analysis.txt'), 'w') as f:
                f.write("Analysis Report\n")
                f.write("===============\n\n")
                f.write(f"General Accuracy: {general_accuracy * 100:.2f}%\n\n")
                f.write(f"Best Guess (Treatment, Accuracy): {best_guess}\n")
                f.write(f"Worst Guess (Treatment, Accuracy): {worst_guess}\n\n")
                f.write("Accuracy per Treatment:\n")
                for label, acc in class_accuracies:
                    f.write(f"{label}: {acc * 100:.2f}%\n")


if __name__ == "__main__":
    # Ask for the train and test folders
    root = tk.Tk()
    root.withdraw()
    labels_encoding = ['0', '1', '2', '3']
    treatments_indices = {'2': 'LD', '0': '2lux', '1': '5lux', '3': 'LL'}
    print("Select the train folder")
    # train_folder = filedialog.askdirectory()

    print("Select the test folder")
    test_folder = filedialog.askdirectory()

    # Delete files in test folder if found same file in the train folder
    # supporting_functions.delete_common_files(train_folder, test_folder)

    # Load model
    model_path = r'accuracy_97.37_num_files_362_batch_size_30_30_parts_mfcc-h5py\best_model.pth'
    evaluator = ModelEvaluator(model_path)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels_encoding)
    data_loader = CustomDataLoader(test_folder, num_files_per_treatment=100)
    data_loader.split_data_files(diagnostic_mode=True)
    test_dataset = CustomDataset(data_loader.test_files)
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True, num_workers=12)
    test_mean, test_std = compute_mean_std(test_loader)
    ready_set = CustomDataset(data_loader.test_files, labels=label_encoder, mean=test_mean, std=test_std)
    ready_loader = DataLoader(ready_set, batch_size=30, shuffle=False, num_workers=12)  # DataLoader for ready_set

    # Evaluate
    confusion_mat = evaluator.evaluate(ready_loader)

    # Plot confusion matrix
    evaluator.save_and_show_results(confusion_mat, treatments_indices, test_folder)


