import os
import random
import gzip
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
from collections import deque
logging.basicConfig(level=logging.INFO)


class DataLoader:

    def __init__(self, folder_path, num_files_per_treatment=200, batch_size=32, min_file_size=7000000, max_file_size=8000000, file_extension='.gz', test_size=0.2):
        self.folder_path = folder_path
        self.num_files_per_treatment = num_files_per_treatment
        self.batch_size = batch_size
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.file_extension = file_extension
        self.test_size = test_size
        self.train_files = {}  # Dictionary to store treatment-wise training files
        self.test_files = []

    @staticmethod
    def load_data(data_path):
        with gzip.open(data_path, 'rt') as fp:
            data = json.load(fp)
        return np.array(data["mfcc"]), np.array(data["labels"])

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

        for treatment, files in treatment_files.items():
            train, test = train_test_split(files, test_size=self.test_size)
            self.train_files[treatment] = train
            self.test_files.extend(test)

    def data_generator(self, batches_per_epoch=4):
        treatments = list(self.train_files.keys())
        num_treatments = len(treatments)
        files_per_treatment_in_batch = self.batch_size // num_treatments

        while True:  # Infinite loop to keep the generator alive across multiple epochs

            if not all(
                    len(self.train_files[treatment]) >= files_per_treatment_in_batch for treatment in treatments):
                break

            X_accumulate = []
            y_accumulate = []
            batch_files = []

            for treatment in treatments:
                selected_files = self.train_files[treatment][:files_per_treatment_in_batch]
                batch_files.extend(selected_files)
                # Remove the selected files for the next iteration
                self.train_files[treatment] = self.train_files[treatment][files_per_treatment_in_batch:]

            np.random.shuffle(batch_files)  # Shuffle the combined batch files

            for file_path in batch_files:
                try:
                    X, y = self.load_data(file_path)
                    X_accumulate.extend(X)
                    y_accumulate.extend(y)

                    while len(X_accumulate) >= self.batch_size:
                        yield np.array(X_accumulate[:self.batch_size]), np.array(y_accumulate[:self.batch_size])
                        X_accumulate = X_accumulate[self.batch_size:]
                        y_accumulate = y_accumulate[self.batch_size:]

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")


class ModelTrainer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.cumulative_sum = 0
        self.cumulative_squared_sum = 0
        self.num_samples = 0

        # Initialization for the input shape.
        X_sample, y_sample = next(data_loader.data_generator())
        X_sample = np.expand_dims(X_sample, axis=-1)
        input_shape = (X_sample.shape[1], X_sample.shape[2], 1)

        self.model = self.build_model(input_shape=input_shape)

    @staticmethod
    def build_model(input_shape):
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
            keras.layers.Dropout(0.1),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
            keras.layers.Dropout(0.1),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(4, activation='softmax')  # Change to 4 to match your number of treatments
        ])

        optimiser = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimiser,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def save_model_if_good(self, correct_preds, total_preds, save_dir='saved_models'):
        accuracy = correct_preds / total_preds
        if accuracy >= 0.9:
            os.makedirs(save_dir, exist_ok=True)
            model_file_path = os.path.join(save_dir, f'model_accuracy_{accuracy:.2f}.h5')
            self.model.save(model_file_path)
            print(f'Model saved at {model_file_path}')

    def train_model(self, epochs=50):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Reset batch accuracy trackers and predictions counters for each epoch
            batch_accuracies = []
            total_correct_preds = 0
            total_preds = 0

            X_batch, y_batch = next(self.data_loader.data_generator())

            X_batch = np.expand_dims(X_batch, axis=-1)

            # Splitting data for training and validation
            X_train, X_valid, y_train, y_valid = train_test_split(X_batch, y_batch, test_size=0.2)

            # Update statistics only with training data
            self.update_stats(X_train)

            # Normalize training data
            X_train_norm = self.normalize_data(X_train)
            X_train_norm = np.expand_dims(X_train_norm, axis=-1)

            # Normalize validation data using the same statistics from training
            X_valid_norm = self.normalize_data(X_valid)
            X_valid_norm = np.expand_dims(X_valid_norm, axis=-1)

            # Train model on the batch
            history = self.model.train_on_batch(X_train_norm, y_train)

            # Evaluate model's performance on this batch's validation data
            val_accuracy = self.model.evaluate(X_valid_norm, y_valid, verbose=0)[1]
            batch_accuracies.append(val_accuracy)

            # Update counters for saving decision later
            y_pred = self.model.predict(X_valid_norm)
            y_pred_classes = np.argmax(y_pred, axis=1)
            total_correct_preds += np.sum(y_valid == y_pred_classes)
            total_preds += len(y_valid)

            print(f"Batch validation accuracy: {val_accuracy:.2f}")

            # After processing all batches for this epoch, print average accuracy for this epoch
            epoch_accuracy = np.mean(batch_accuracies)
            print(f"Average validation accuracy after epoch {epoch + 1}: {epoch_accuracy:.2f}")

            # Save model after all epochs based on the entire epoch's performance
            self.save_model_if_good(correct_preds=total_correct_preds, total_preds=total_preds)

    def update_stats(self, data):
        # Update cumulative statistics
        self.cumulative_sum += np.sum(data, axis=(0, 1))
        self.cumulative_squared_sum += np.sum(data ** 2, axis=(0, 1))
        self.num_samples += data.shape[0]

        self.mean = self.cumulative_sum / self.num_samples
        self.variance = (self.cumulative_squared_sum / self.num_samples) - (self.mean ** 2)
        self.std = np.sqrt(self.variance + 1e-7)

    def normalize_data(self, data):
        return (data - self.mean) / self.std

    def test_model(self):
        correct_preds = 0
        total_preds = 0

        for file_path in self.data_loader.test_files:
            try:
                X_test, y_test = self.data_loader.load_data(file_path)
                X_test = self.normalize_data(X_test)
                X_test = np.expand_dims(X_test, axis=-1)

                y_pred = self.model.predict(X_test)
                y_pred_classes = np.argmax(y_pred, axis=1)

                correct_preds += np.sum(y_test == y_pred_classes)
                total_preds += len(y_test)
            except Exception as e:
                logging.warning(f"Error processing file {file_path}: {e}")

        accuracy = correct_preds / total_preds
        print(f"Test accuracy: {accuracy:.2f}")
        self.save_model_if_good(correct_preds, total_preds)

