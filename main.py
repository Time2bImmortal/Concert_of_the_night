import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import os
import gzip
import random
from tkinter import filedialog
import tkinter as tk
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from model_testing import ModelEvaluator
import Preprocessing
import pickle
from playsound import playsound
audio_start = "/audio/begin.wav"
audio_galaxy = '/audio/galaxy.wav'
class DataLoader:
    def __init__(self, folder_path, num_files_per_treatment=128):
        self.folder_path = folder_path
        self.num_files_per_treatment = num_files_per_treatment
        self.X, self.y = self.load_data_from_folder()

    @staticmethod
    def load_data(data_path):
        with gzip.open(data_path, 'rt') as fp:
            data = json.load(fp)

        X = np.array(data["mfcc"])
        y = np.array(data["labels"])
        return X, y

    def load_data_from_folder(self, min_file_size=7000000, max_file_size=8000000):
        X, y = [], []
        correct_shape = None

        treatments = os.listdir(self.folder_path)
        for treatment in treatments:
            treatment_path = os.path.join(self.folder_path, treatment)
            if os.path.isdir(treatment_path):
                all_files = [f for f in os.listdir(treatment_path) if f.endswith('.gz')]
                selected_files = random.sample(all_files, self.num_files_per_treatment)
                print(f"Processing treatment: {treatment}")
                for file in tqdm(selected_files, desc="Loading files"):
                    file_path = os.path.join(treatment_path, file)
                    file_size = os.path.getsize(file_path)
                    if min_file_size <= file_size <= max_file_size:
                        X_temp, y_temp = self.load_data(file_path)
                        if correct_shape is None:
                            correct_shape = X_temp.shape
                        if X_temp.shape == correct_shape:
                            X.append(X_temp)
                            y.append(y_temp)

        if X:
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y, axis=0)
            print("\nData successfully loaded!")
        else:
            print("No valid data loaded!")

        return X, y


class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.X = np.expand_dims(X, axis=-1)
        self.y = y
        self.model = self.build_model()

    def build_model(self):

        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(self.X.shape[1], self.X.shape[2], self.X.shape[3])),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Conv2D(64, (2, 2), activation='relu'),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Conv2D(128, (2, 2), activation='relu'),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(4, activation='softmax')  # Change 10 to 4 for 4 classes
        ])

        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def save_model_if_good(self, correct_preds, total_preds, save_dir='saved_models'):
        accuracy = correct_preds / total_preds
        if accuracy >= 0.8:
            os.makedirs(save_dir, exist_ok=True)
            model_file_path = os.path.join(save_dir, f'shape77520.h5')
            self.model.save(model_file_path)
            print(f'Model saved at {model_file_path}')

    def train_model(self, batch_size=8, epochs=64):

        X_train_val, X_test, y_train_val, y_test = train_test_split(self.X, self.y, test_size=0.15)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15)

        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        with open('train_data_std_scaler.pkl', 'wb') as f:
            pickle.dump({'mean': mean, 'std': std}, f)

        X_train -= mean
        X_train /= std

        X_val -= mean
        X_val /= std

        X_test -= mean
        X_test /= std

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_val = label_encoder.transform(y_val)
        y_test = label_encoder.transform(y_test)

        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size,
                                     epochs=epochs)


        self.predict_file_classes(X_test, y_test, segments_per_file=1)

    def predict_file_classes(self, X, y, segments_per_file=30):
        # Predict classes for each segment
        y_pred = self.model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)

        file_preds = []
        for i in range(0, len(y_pred_classes), segments_per_file):
            segment_preds = y_pred_classes[i:i + segments_per_file]
            # Majority voting
            file_pred = np.bincount(segment_preds).argmax()
            file_preds.append(file_pred)

        file_preds = np.array(file_preds)

        file_labels = []
        for i in range(0, len(y), segments_per_file):
            file_label = y[i]
            file_labels.append(file_label)
        file_labels = np.array(file_labels)

        correct_preds = np.sum(file_labels == file_preds)
        print(f'Correct predictions: {correct_preds} out of {len(file_labels)}')

        total_preds = len(file_labels)
        self.save_model_if_good(correct_preds, total_preds)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()
    data_loader = DataLoader(folder_path)
    trainer = ModelTrainer(data_loader.X, data_loader.y)
    trainer.train_model()
    test_folder = "G:/one_mfcc"
    model_path = 'saved_models/shape77520.h5'  # specify the correct model path
    evaluator = ModelEvaluator(model_path, test_folder)
    data_loader = DataLoader(test_folder, 64)  # Please make sure DataLoader class is defined somewhere

    # Evaluate
    confusion_mat = evaluator.evaluate(data_loader.X, data_loader.y)
    treatments_indices = {'2lux': 0, '5lux': 1, 'LD': 2, 'LL': 3}
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(confusion_mat, treatments_indices, "one_mfcc")

    # src_directory = "G:\Stridulation syllable patterns"
    # processor = Preprocessing.AudioProcessor('mfcc', src_directory)
    # processor.run()
