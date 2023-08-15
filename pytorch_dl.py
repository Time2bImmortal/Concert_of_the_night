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


class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.label_encoder = LabelEncoder()
        self.mfccs_per_file = 30

    def __len__(self):
        return len(self.file_paths) * self.mfccs_per_file

    def __getitem__(self, idx):
        # Calculate the file index and the mfcc index within that file
        file_idx = idx // self.mfccs_per_file
        mfcc_idx = idx % self.mfccs_per_file

        with gzip.open(self.file_paths[file_idx], 'rt') as fp:
            data = json.load(fp)

        # Encoding the labels from string to integers. This assumes that the 'labels' key in the data is a list.
        label = self.label_encoder.fit_transform([data["labels"][0]])[0]
        return torch.tensor(data["mfcc"][mfcc_idx], dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class CustomDataLoader:
    def __init__(self, folder_path, num_files_per_treatment=100, batch_size=30, min_file_size=7000000, max_file_size=8000000, file_extension='.gz', test_size=0.1):
        self.folder_path = folder_path
        self.num_files_per_treatment = num_files_per_treatment
        self.batch_size = batch_size
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.file_extension = file_extension
        self.test_size = test_size
        self.train_files = []
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
            train, test = train_test_split(files, test_size=self.test_size)
            self.train_files.extend(train)
            self.test_files.extend(test)

        # Shuffle the train files
        random.shuffle(self.train_files)

        # Splitting the train set further to obtain a validation set, for instance, 80% train and 20% validation
        self.train_files, self.val_files = train_test_split(self.train_files, test_size=0.2)

    def get_data_loader(self, split='train'):
        if split == 'train':
            dataset = CustomDataset(self.train_files)
        elif split == 'val':
            dataset = CustomDataset(self.val_files)
        else:  # test
            dataset = CustomDataset(self.test_files)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


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

if __name__ == '__main__':

    folder_path = filedialog.askdirectory()
    print(f"The path: {folder_path} is going to be treated now")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'
    print(f"Using: {device} device")

    # Load data
    print("loading data...")
    data_loader = CustomDataLoader(folder_path, batch_size=10)
    data_loader.split_data_files()
    train_loader = data_loader.get_data_loader('train')
    test_loader = data_loader.get_data_loader('test')
    print("data was loaded successfully")
    # Model, Loss and Optimizer
    print("Model is initiated...")
    model = MFCC_CNN()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training settings
    n_epochs = 50  # Feel free to adjust this value
    best_accuracy = 0.95  # Save model if accuracy > this value
    print("Starting epochs")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()

        train_accuracy = 100. * correct_train / total_train
        print(
            f"Epoch {epoch + 1}/{n_epochs}\t Training loss: {total_loss / (batch_idx + 1):.4f} | Training accuracy: {train_accuracy:.2f}%")

        # Validation on the test set
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                total_test += target.size(0)
                correct_test += predicted.eq(target).sum().item()

        test_accuracy = 100. * correct_test / total_test
        print(f"Epoch {epoch + 1}/{n_epochs}\t Test accuracy: {test_accuracy:.2f}%")

        # Save model if it reaches desired accuracy
        if test_accuracy >= best_accuracy:
            print(f"Accuracy of {test_accuracy:.2f}% reached. Saving model...")
            torch.save(model.state_dict(), "best_model.pth")

    # Final results on the test set
    print(f"Final test accuracy after {n_epochs} epochs: {test_accuracy:.2f}%")
