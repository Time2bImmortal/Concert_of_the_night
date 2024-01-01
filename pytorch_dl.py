import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tkinter import filedialog
import random
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import gc
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
from collections import defaultdict
from itertools import combinations


def plot_confusion_matrix(cm, result_dir, title='Confusion matrix'):
    # The treatments_indices dictionary
    treatments_indices = {'2': 'LD', '0': '2lux', '1': '5lux', '3': 'LL'}

    # Custom ordering function for the keys
    order = {'2': 0, '0': 1, '1': 2, '3': 3}
    sorted_keys = sorted(treatments_indices.keys(), key=lambda x: order[x])

    labels = [treatments_indices[key] for key in sorted_keys]

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())  # Explicitly passing the current axes (ax) to the plot method
    plt.title(title)

    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()


def compute_mean_std(loader):
    mean = 0.0
    var = 0.0
    nb_samples = 0

    # loop through each batch in the loader
    for idx, (data, _) in enumerate(loader):
        if idx % 100 == 0:  # Print every 100 batches
            print(f"Processing batch {idx + 1}...")
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        var += data.var(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)

    return mean, std


# class CustomDataset(Dataset):
#     def __init__(self, file_paths, labels=None, mean=None, std=None, shuffle_labels=False):
#         self.file_paths = file_paths
#         self.mean = mean
#         self.std = std
#         self.label_encoder = labels
#         self.shuffle_labels = shuffle_labels
#
#         # Extract all labels first if shuffle_labels is True
#         if self.shuffle_labels:
#             all_labels = []
#             for path in self.file_paths:
#                 with h5py.File(path, 'r') as h5_file:
#                     all_labels.extend(h5_file['labels'][:])
#             random.shuffle(all_labels)
#             self.shuffled_labels = all_labels
#
#         # Dynamically determine the number of MFCCs using the first file
#         with h5py.File(self.file_paths[0], 'r') as h5_file:
#             mfcc_data = h5_file['mfcc'][:]
#         self.mfccs_per_file = len(mfcc_data)
#
#     def __len__(self):
#         return len(self.file_paths) * self.mfccs_per_file
#
#     def __getitem__(self, idx):
#         # Calculate the file index and the mfcc index within that file
#         file_idx = idx // self.mfccs_per_file
#         mfcc_idx = idx % self.mfccs_per_file
#         file_path = self.file_paths[file_idx]
#
#         try:
#             with h5py.File(file_path, 'r') as h5_file:
#                 mfcc_data = h5_file['mfcc'][mfcc_idx]
#
#                 if self.shuffle_labels:
#                     label_data = self.shuffled_labels[idx]
#                 else:
#                     label_data = h5_file['labels'][0]
#
#             if isinstance(label_data, bytes):
#                 label_data = label_data.decode('utf-8')
#
#             # Transform the label using the already fitted label_encoder
#             if self.label_encoder is not None:
#                 label = self.label_encoder.transform([label_data])[0]
#             else:
#                 label = label_data
#
#             tensor_mfcc = torch.tensor(mfcc_data, dtype=torch.float32)
#
#             if self.mean is not None and self.std is not None:
#                 # Convert mean and std to tensors if they are not
#                 if not torch.is_tensor(self.mean):
#                     self.mean = torch.tensor(self.mean, dtype=torch.float32)
#                 if not torch.is_tensor(self.std):
#                     self.std = torch.tensor(self.std, dtype=torch.float32)
#
#                 tensor_mfcc = (tensor_mfcc - self.mean[:, None]) / self.std[:, None]
#
#             tensor_label = torch.tensor(label, dtype=torch.long)
#             return tensor_mfcc, tensor_label
#
#         except Exception as e:
#             print(f"Error processing file at path {file_path}: {e}")
#             # If you have a default or dummy tensor to return in case of error, do that here.
#             # Otherwise, raise the exception to stop the process.
#             raise e
class CustomDataset(Dataset):
    def __init__(self, file_paths, feature_name, labels=None, mean=None, std=None, shuffle_labels=False):
        self.file_paths = file_paths
        self.feature_name = feature_name  # Added feature name
        self.mean = mean
        self.std = std
        self.label_encoder = labels
        self.shuffle_labels = shuffle_labels
        self.valid_indices = []

        # Scan through the files to find valid (non-None) features and their corresponding indices
        for file_idx, path in enumerate(self.file_paths):
            with h5py.File(path, 'r') as h5_file:
                for feature_idx, feature in enumerate(h5_file[self.feature_name]):
                    if feature is not None:  # Only keep indices for non-None features
                        self.valid_indices.append((file_idx, feature_idx))

        # If labels need to be shuffled, do so
        if self.shuffle_labels:
            random.shuffle(self.valid_indices)  # Note: this shuffles the features too

        self.feature_shape = self.get_feature_shape()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_idx, feature_idx = self.valid_indices[idx]
        file_path = self.file_paths[file_idx]

        with h5py.File(file_path, 'r') as h5_file:
            feature_data = h5_file[self.feature_name][feature_idx]
            label_data = h5_file['labels'][feature_idx]

            if isinstance(label_data, bytes):
                label_data = label_data.decode('utf-8')

            if self.label_encoder is not None:
                label = self.label_encoder.transform([label_data])[0]
            else:
                label = label_data

            tensor_feature = torch.tensor(feature_data, dtype=torch.float32)

            if self.mean is not None and self.std is not None:
                if not torch.is_tensor(self.mean):
                    self.mean = torch.tensor(self.mean, dtype=torch.float32)
                if not torch.is_tensor(self.std):
                    self.std = torch.tensor(self.std, dtype=torch.float32)

                tensor_feature = (tensor_feature - self.mean[:, None]) / self.std[:, None]

            tensor_label = torch.tensor(label, dtype=torch.long)

            return tensor_feature, tensor_label

    def get_feature_shape(self):
        # Retrieve the shape of the first valid feature
        for file_idx, feature_idx in self.valid_indices:
            with h5py.File(self.file_paths[file_idx], 'r') as h5_file:
                feature_data = h5_file[self.feature_name][feature_idx]
                return feature_data.shape
class CustomDataLoaderWithoutSubjects:
    def __init__(self, folder_path, min_file_size=10000000, max_file_size=50000000,
                 file_extension='.h5', test_size=0.1, valid_size=0.2, num_files_per_treatment=185):
        self.folder_path = folder_path
        self.num_files_per_treatment = num_files_per_treatment
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.file_extension = file_extension
        self.test_size = test_size
        self.valid_size = valid_size
        self.train_files = []
        self.val_files = []
        self.test_files = []

    def _get_filtered_files(self):
        return self._get_files_without_subjects()

    def _get_files_without_subjects(self):
        treatment_files = {}
        treatments = os.listdir(self.folder_path)
        for treatment in treatments:
            treatment_path = os.path.join(self.folder_path, treatment)
            if os.path.isdir(treatment_path):
                valid_files = [f for f in os.listdir(treatment_path) if
                               f.endswith(self.file_extension) and self.min_file_size <= os.path.getsize(
                                   os.path.join(treatment_path, f)) <= self.max_file_size]
                treatment_files[treatment] = [os.path.join(treatment_path, f) for f in
                                              valid_files[:self.num_files_per_treatment]]
        return treatment_files

    def split_data_files(self, diagnostic_mode=False):
        treatment_files = self._get_filtered_files()

        if diagnostic_mode:
            # All files are used for testing if test flag is True
            for treatment, files in treatment_files.items():
                self.test_files.extend(files)
        else:
            all_train_files = []
            all_test_files = []
            for treatment, files in treatment_files.items():
                logging.info(f"{treatment}: {len(files)} files")
                random.shuffle(files)  # Shuffle the files before splitting
                train, test_data = train_test_split(files, test_size=self.test_size)
                all_train_files.extend(train)
                all_test_files.extend(test_data)
            self.train_files, self.val_files = train_test_split(all_train_files, test_size=self.valid_size)
            self.test_files = all_test_files

        logging.info(f"Train files: {len(self.train_files)}")
        logging.info(f"Validation files: {len(self.val_files)}")
        logging.info(f"Test files: {len(self.test_files)}")


class CustomDataLoaderWithSubjects:
    def __init__(self, folder_path, result_dir, min_file_size=0, max_file_size=250000000,
                 file_extension='.h5', num_files_per_subject=9, num_test_subjects=2, num_train_valid_subjects=7, num_folds=1):
        self.folder_path = folder_path
        self.result_dir = result_dir
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.file_extension = file_extension
        self.num_folds = num_folds
        self.current_fold = 0
        self.num_files_per_subject = num_files_per_subject
        self.num_test_subjects = num_test_subjects
        self.num_train_valid_subjects = num_train_valid_subjects
        self.train_files = []
        self.val_files = []
        self.test_files = []
        self._initialize_subject_splits()

    def _initialize_subject_splits(self):
        self.train_valid_subjects_dict = defaultdict(list)
        self.test_subjects_dict = defaultdict(list)
        self.train_valid_combinations = defaultdict(list)

        treatments = os.listdir(self.folder_path)

        for treatment in treatments:
            treatment_path = os.path.join(self.folder_path, treatment)
            subjects = os.listdir(treatment_path)
            random.shuffle(subjects)

            subjects = subjects[:9]
            train_valid_subjects, test_subjects = train_test_split(subjects, test_size=self.num_test_subjects)

            self.test_subjects_dict[treatment] = test_subjects

            all_permutations = [random.sample(train_valid_subjects, len(train_valid_subjects)) for _ in range(self.num_folds)]
            self.train_valid_combinations[treatment] = all_permutations
        for treatment, subjects in self.test_subjects_dict.items():
            treatment_path = os.path.join(self.folder_path, treatment)
            for subject in subjects:
                subject_path = os.path.join(treatment_path, subject)
                self.test_files.extend(self._get_valid_files_from_subject(subject_path))

    def _get_valid_files_from_subject(self, subject_path):
        valid_files = [f for f in os.listdir(subject_path) if
                       f.endswith(self.file_extension) and self.min_file_size <= os.path.getsize(
                           os.path.join(subject_path, f)) <= self.max_file_size]
        chosen_files = random.sample(valid_files, self.num_files_per_subject)
        return [os.path.join(subject_path, f) for f in chosen_files]

    def split_data_files(self):
        self.train_files = []
        self.val_files = []

        for treatment in self.train_valid_combinations:
            treatment_path = os.path.join(self.folder_path, treatment)
            train_subjects = self.train_valid_combinations[treatment][self.current_fold][:5]
            validation_subjects = self.train_valid_combinations[treatment][self.current_fold][5:]

            for subject in train_subjects:
                subject_path = os.path.join(treatment_path, subject)
                self.train_files.extend(self._get_valid_files_from_subject(subject_path))

            for subject in validation_subjects:
                subject_path = os.path.join(treatment_path, subject)
                self.val_files.extend(self._get_valid_files_from_subject(subject_path))
        logging.info(f"Training files count: {len(self.train_files)}")
        logging.info(f"Validation files count: {len(self.val_files)}")
        logging.info(f"Test files count: {len(self.test_files)}")
        self._write_subjects_to_file()

    def next_fold(self):
        self.current_fold = self.current_fold + 1
        self.split_data_files()

    def _write_subjects_to_file(self):
        with open(os.path.join(self.result_dir, 'subjects_log.txt'), 'a') as log_file:
            log_file.write(f"Fold {self.current_fold + 1}/{self.num_folds}\n")
            log_file.write("Test subjects:\n")
            for treatment, subjects in self.test_subjects_dict.items():
                short_subjects = [subject[-8:] for subject in subjects]
                log_file.write(f"{treatment}: {', '.join(short_subjects)}\n")

            for treatment in self.train_valid_combinations:
                # Using directly determined subjects in split_data_files
                train_subjects = [subject[-8:] for subject in
                                  self.train_valid_combinations[treatment][self.current_fold][:5]]
                validation_subjects = [subject[-8:] for subject in
                                       self.train_valid_combinations[treatment][self.current_fold][5:]]

                log_file.write(f"Training subjects for treatment {treatment}: {', '.join(train_subjects)}\n")
                log_file.write(f"Validation subjects for treatment {treatment}: {', '.join(validation_subjects)}\n")
            log_file.write("=" * 40 + "\n")

class MFCC_CRNN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(MFCC_CRNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 4), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout_conv1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 2), stride=(2, 2), padding=(2, 4))
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout_conv2 = nn.Dropout(0.5)

        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)

        self.lstm_input_dim = self._get_conv_output((1, 128, 1723))

        # LSTM
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 4)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.dropout_conv1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout_conv2(self.pool(F.relu(self.bn2(self.conv2(x)))))

        # Prepare data for LSTM
        x = x.view(x.size(0), x.size(2), -1)

        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last LSTM output

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    # Helper function to calculate the LSTM input size
    def _get_conv_output(self, shape):
        bs = 1
        input_tensor = torch.rand(bs, *shape)
        output_feat = self._forward_features(input_tensor)
        n_size = output_feat.data.size(2)
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), x.size(2), -1)
        return x
class MFCC_CNN(nn.Module):
    def __init__(self, transformer_flag=False):
        super(MFCC_CNN, self).__init__()
        self.transformer_flag = transformer_flag


        self.conv1 = nn.Conv2d(1, 32, kernel_size=(39, 6), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout_conv1 = nn.Dropout(0.4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(6, 3), stride=(2, 2), padding=(2, 4))
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout_conv2 = nn.Dropout(0.4)

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 10), stride=(2, 2), padding=(2, 4))
        # self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)

        # Compute the output size after convolution and pooling layers to use in the FC layer
        self.fc_input_dim = self._get_conv_output((1, 39, 2584))

        # Fully connected layers
        self.fc1 = nn.Linear(10432, 64)
        self.fc2 = nn.Linear(64, 4)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.dropout_conv1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout_conv2(self.pool(F.relu(self.bn2(self.conv2(x)))))
        # x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        if self.transformer_flag:
            # Convert the output to shape suitable for transformer: [sequence_length, batch_size, d_model]
            x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)
        else:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

        return x

    # Helper function to calculate the number of units in the Fully Connected layer
    def _get_conv_output(self, shape):
        bs = 1
        input_tensor = torch.rand(bs, *shape)
        output_feat = self._forward_features(input_tensor)
        n_size = output_feat.data.size(1)  # Instead of view, we directly get size since it's already flattened by GAP.
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor
        x = x.view(x.size(0), -1)
        return x


class CombinedModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(CombinedModel, self).__init__()

        self.cnn = MFCC_CNN(transformer_flag=True)
        self.transformer_block = TransformerBlock(d_model, nhead, num_layers)

        # Depending on your final reshaped size from CNN and the transformer's output size, adjust this value
        self.fc_input_dim = d_model  # This might need adjustment
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.cnn(x)
        print(f"After CNN: {x.shape}")
        x = self.transformer_block(x)
        print(f"After Transformer: {x.shape}")

        # Possibly average along the sequence length dimension after the transformer
        x = torch.mean(x, dim=0)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device):
        logging.info(f"Training initialized ...")
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0  # counter for early stopping
        self.patience = 5

    def train_epoch(self):
        print_interval = len(self.train_loader) // 5  # print 10 times per epoch
        self.model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        start_time = time.time()
        last_print_time = start_time  # Initialize the last print time to the start time
        saved=False
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # if not saved:  # Only save once
                # data = add_gaussian_noise(data)  # This function now also saves the data
            #     saved = True  # Update the flag
            # else:
            #     data = add_gaussian_noise_without_saving(data)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.loss_fn(outputs, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()

            # Printing progress within the epoch
            if (batch_idx + 1) % print_interval == 0:
                elapsed_time = time.time() - last_print_time
                print(
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} - Loss: {loss.item():.4f} - Time elapsed: {elapsed_time:.2f}s")
                last_print_time = time.time()  # Update the last print time

        accuracy = 100. * correct_train / total_train

        return total_loss / (batch_idx + 1), accuracy

    def evaluate(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_true_labels = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(target.cpu().numpy())
        accuracy = 100. * sum(np.array(all_predictions) == np.array(all_true_labels)) / len(all_true_labels)
        return accuracy, all_predictions, all_true_labels

    def check_for_pause(self, epoch, pause_path, checkpoint_path):
        if os.path.exists(pause_path):
            self.save_checkpoint(epoch, checkpoint_path)
            print(
                f"Paused and saved at epoch {epoch}. To continue, remove the pause.txt file and start training again.")
            return True
        return False

    def evaluate_test_set(self, result_dir, folder_name, test_loader):
        test_accuracy, test_predictions, test_true_labels = self.evaluate(test_loader)
        cm = confusion_matrix(test_true_labels, test_predictions)
        plot_confusion_matrix(cm, result_dir=result_dir)
        print(f"Final test accuracy: {test_accuracy:.2f}%")
        self.plot_learning_curve(result_dir)
        self.save_model(test_accuracy, folder_name, result_dir)
        return test_accuracy, test_predictions, test_true_labels

    def train(self, n_epochs, best_accuracy, folder_name, directory, result_dir):
        pause_path = os.path.join(directory, "pause.txt")
        checkpoint_path = os.path.join(directory, "saved_training_step.pth")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Check for saved_training_step at the start
        if os.path.exists(checkpoint_path):
            logging.info(
                "A model already exists, let's continue with it (move it in another directory if you want to keep it"
                "or erase it, I don't care, just do something with your pathetic life")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.train_losses = checkpoint.get('train_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
        else:
            start_epoch = 0
        logging.info("Epochs loop starting...")
        for epoch in range(start_epoch, n_epochs):
            train_loss, train_accuracy = self.train_epoch()
            val_accuracy = self.evaluate(self.val_loader)[0]
            print(f"Epoch {epoch + 1}/{n_epochs}\t Training loss: {train_loss:.4f} | Training accuracy: "
                  f"{train_accuracy:.2f}% | Validation accuracy: {val_accuracy:.2f}%")

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            if val_accuracy > max(self.val_accuracies, default=0):
                self.best_model_state = self.model.state_dict()
            self.val_accuracies.append(val_accuracy)
            if float(val_accuracy) >= best_accuracy:
                print(f"Accuracy of {val_accuracy:.2f}% reached. Saving model...")
                self.evaluate_test_set(result_dir, folder_name)

            if self.check_for_pause(epoch, pause_path, checkpoint_path):
                break

    def save_checkpoint(self, epoch, path):
        """Save current training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }

        torch.save(checkpoint, path)
        print(f"Saved training state at epoch {epoch} to {path}.")

    def plot_learning_curve(self, result_dir):
        epochs = range(1, len(self.train_losses) + 1)

        # Adjust x-ticks
        x_ticks = list(range(1, len(self.train_losses) + 1, 5))
        if len(self.train_losses) not in x_ticks:
            x_ticks.append(len(self.train_losses))

        # Plot training and validation accuracy values
        plt.figure()
        plt.plot(epochs, self.train_accuracies, 'b', label='Training accuracy')
        plt.plot(epochs, self.val_accuracies, 'r', label='Validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.xticks(x_ticks)
        plt.savefig(os.path.join(result_dir, "learning_curve_accuracy.png"))
        plt.close()

        # Plot training loss values
        plt.figure()
        plt.plot(epochs, self.train_losses, 'b', label='Training loss')
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.xticks(x_ticks)
        plt.savefig(os.path.join(result_dir, "learning_curve_loss.png"))
        plt.close()

    def save_model(self, accuracy, folder_name, result_dir):

        # Create the directory name based on the given parameters
        dir_name = f"accuracy_{accuracy:.2f}_{folder_name}_"
        save_dir = os.path.join(result_dir, dir_name)
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Path to save the model
        model_path = os.path.join(save_dir, 'best_model.pth')

        # Save the model
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    def get_best_model_state(self):
        return self.best_model_state


def clear_memory():
    # Python's garbage collector
    gc.collect()

    # PyTorch's memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

