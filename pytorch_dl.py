import os
import gzip
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import tkinter
from tkinter import filedialog
import random
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import gc
import time
import h5py
def compute_mean_std(loader):
    print("compute_mean_std function called...")

    mean = 0.0
    var = 0.0
    nb_samples = 0

    # loop through each batch in the loader
    for idx, (data, _) in enumerate(loader):
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


class CustomDataset(Dataset):
    def __init__(self, file_paths, labels = None, mean=None, std=None):
        self.file_paths = file_paths
        self.mean = mean
        self.std = std
        # Initialize and fit the label encoder using provided labels
        self.label_encoder = labels

        # Dynamically determine the number of MFCCs using the first file
        with h5py.File(self.file_paths[0], 'r') as h5_file:
            mfcc_data = h5_file['mfcc'][:]
        self.mfccs_per_file = len(mfcc_data)

    def __len__(self):
        return len(self.file_paths) * self.mfccs_per_file

    def __getitem__(self, idx):
        # Calculate the file index and the mfcc index within that file
        file_idx = idx // self.mfccs_per_file
        mfcc_idx = idx % self.mfccs_per_file

        with h5py.File(self.file_paths[file_idx], 'r') as h5_file:
            mfcc_data = h5_file['mfcc'][mfcc_idx]
            label_data = h5_file['labels'][0]  # assuming 'labels' dataset exists and you need the first element

        # Transform the label using the already fitted label_encoder
        if self.label_encoder is not None:
            label = self.label_encoder.transform([label_data])[0]
        else:
            label = label_data

        tensor_mfcc = torch.tensor(mfcc_data, dtype=torch.float32)

        if self.mean is not None and self.std is not None:
            # Convert mean and std to tensors if they are not
            if not torch.is_tensor(self.mean):
                self.mean = torch.tensor(self.mean, dtype=torch.float32)
            if not torch.is_tensor(self.std):
                self.std = torch.tensor(self.std, dtype=torch.float32)

            tensor_mfcc = (tensor_mfcc - self.mean[:, None]) / self.std[:, None]


        tensor_label = torch.tensor(label, dtype=torch.long)
        return tensor_mfcc, tensor_label


class CustomDataLoader:
    def __init__(self, folder_path, num_files_per_treatment=300, min_file_size=3600000, max_file_size=4600000, file_extension='.h5', test_size=0.1):
        self.folder_path = folder_path
        self.num_files_per_treatment = num_files_per_treatment
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.file_extension = file_extension
        self.test_size = test_size
        self.train_files = []
        self.val_files = []
        self.test_files = []

    def _get_filtered_files(self):
        treatment_files = {}
        treatments = os.listdir(self.folder_path)
        for treatment in treatments:
            treatment_path = os.path.join(self.folder_path, treatment)
            if os.path.isdir(treatment_path):
                valid_files = [f for f in os.listdir(treatment_path) if f.endswith(self.file_extension) and self.min_file_size <= os.path.getsize(os.path.join(treatment_path, f)) <= self.max_file_size]
                treatment_files[treatment] = [os.path.join(treatment_path, f) for f in valid_files[:self.num_files_per_treatment]]
        return treatment_files

    def split_data_files(self):
        treatment_files = self._get_filtered_files()

        for files in treatment_files.values():
            random.shuffle(files)  # Shuffle the files before splitting
            train, test = train_test_split(files, test_size=self.test_size)
            self.train_files.extend(train)
            self.test_files.extend(test)

        # Now, shuffle the train files again (this is optional but ensures randomness after accumulating all train files)
        random.shuffle(self.train_files)
        random.shuffle(self.test_files)

        # Splitting the train set further to obtain a validation set, for instance, 80% train and 20% validation
        self.train_files, self.val_files = train_test_split(self.train_files, test_size=0.2)



class MFCC_CNN(nn.Module):
    def __init__(self):
        super(MFCC_CNN, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 10), stride=1, padding=(2, 5))
        self.bn1 = nn.BatchNorm2d(32) # Batch Normalization after conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 10), stride=1, padding=(2, 5))
        self.bn2 = nn.BatchNorm2d(64) # Batch Normalization after conv2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 10), stride=1, padding=(2, 5))
        self.bn3 = nn.BatchNorm2d(128) # Batch Normalization after conv3

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Compute the output size after convolution and pooling layers to use in the FC layer
        self.fc_input_dim = self._get_conv_output((1, 13, 2584))

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 4)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Pass data through convolution layers with added batch normalization
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the matrix
        x = x.view(x.size(0), -1)

        # Pass data through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    # Helper function to calculate the number of units in the Fully Connected layer
    def _get_conv_output(self, shape):
        bs = 1
        input_tensor = torch.rand(bs, *shape)
        output_feat = self._forward_features(input_tensor)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, loss_fn, device):
        print('Training initialized...')
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self):
        print_interval = len(self.train_loader) // 10  # print 10 times per epoch
        self.model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        start_time = time.time()
        last_print_time = start_time  # Initialize the last print time to the start time
        for batch_idx, (data, target) in enumerate(self.train_loader):

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

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        accuracy = 100. * correct / total
        return accuracy

    def train(self, n_epochs, best_accuracy, num_files, batch_size, folder_name):
        print('Training is starting...')
        for epoch in range(n_epochs):
            train_loss, train_accuracy = self.train_epoch()
            val_accuracy = self.evaluate(self.val_loader)
            print(f"Epoch {epoch + 1}/{n_epochs}\t Training loss: {train_loss:.4f} | Training accuracy: {train_accuracy:.2f}% | Validation accuracy: {val_accuracy:.2f}%")

            if val_accuracy >= best_accuracy:
                print(f"Accuracy of {val_accuracy:.2f}% reached. Saving model...")
                self.save_model(val_accuracy, num_files, batch_size, folder_name)


        # Final results on the test set
        test_accuracy = self.evaluate(self.test_loader)
        print(f"Final test accuracy after {n_epochs} epochs: {test_accuracy:.2f}%")

    def save_model(self, accuracy, num_files, batch_size, folder_name):
        """
        Save the PyTorch model to a specified directory with a specific naming convention.

        :param accuracy: Accuracy of the model
        :param num_files: Number of files used to train the model
        :param batch_size: Batch size used during training
        :param folder_name: Root folder name where the model should be saved
        """

        # Create the directory name based on the given parameters
        dir_name = f"accuracy_{accuracy}_num_files_{num_files}_batch_size_{batch_size}_{folder_name}"

        # Create the directory if it doesn't exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Path to save the model
        model_path = os.path.join(dir_name, 'best_model.pth')

        # Save the model
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")



def clear_memory():
    # Python's garbage collector
    gc.collect()

    # PyTorch's memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('cleaned')
if __name__ == '__main__':
    clear_memory()
    # labels_encoding = ['2lux', '5lux', 'LL', 'LD']
    labels_encoding = ['0', '1', '2', '3']
    BATCH_SIZE = 16
    NUM_FILES = 300
    folder_path = filedialog.askdirectory()
    print(f"The path: {folder_path} is going to be treated now.")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'
    print(f"Using: {device} as processing device.")

    # Load data
    print("Splitting files...")
    data_loader = CustomDataLoader(folder_path, num_files_per_treatment=NUM_FILES)
    data_loader.split_data_files()
    print("Files have been split successfully!")

    label_encoder = LabelEncoder()
    label_encoder.fit(labels_encoding)
    print('Encoder is ready!')

    train_dataset = CustomDataset(data_loader.train_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    train_mean, train_std = compute_mean_std(train_loader)
    print('Normalisation is ready!')

    train_dataset = CustomDataset(data_loader.train_files, labels=label_encoder, mean=train_mean, std=train_std)
    val_dataset = CustomDataset(data_loader.val_files, labels=label_encoder, mean=train_mean, std=train_std)
    test_dataset = CustomDataset(data_loader.test_files, labels=label_encoder, mean=train_mean, std=train_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Data has been loaded and ready to be processed.')

    model = MFCC_CNN()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print('Model is ready.')

    print('Training is starting...')
    trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer, loss_fn, device)
    trainer.train(n_epochs=40, best_accuracy=99, batch_size=BATCH_SIZE, num_files=NUM_FILES, folder_name=os.path.basename(folder_path))


