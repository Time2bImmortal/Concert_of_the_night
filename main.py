import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import os
import gzip
import random
from tkinter import filedialog
import tkinter as tk


class DataLoader:
    def __init__(self, folder_path, num_files_per_treatment=25):
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

    def load_data_from_folder(self):
        X, y = [], []
        correct_shape = None

        for treatment in os.listdir(self.folder_path):
            if treatment:
                treatment_path = os.path.join(self.folder_path, treatment)
                if os.path.isdir(treatment_path):
                    all_files = [f for f in os.listdir(treatment_path) if f.endswith('.gz')]
                    selected_files = random.sample(all_files, self.num_files_per_treatment)
                    for file in selected_files:
                        X_temp, y_temp = self.load_data(os.path.join(treatment_path, file))
                        if correct_shape is None:
                            correct_shape = X_temp.shape
                        if X_temp.shape == correct_shape:
                            X.append(X_temp)
                            y.append(y_temp)

        if X:
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y, axis=0)
            print("Data successfully loaded!")
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
        # model = keras.Sequential([
        #     keras.layers.Flatten(input_shape=(self.X.shape[1], self.X.shape[2])),
        #     keras.layers.Dense(512, activation='relu'),
        #     keras.layers.Dense(256, activation='relu'),
        #     keras.layers.Dense(64, activation='relu'),
        #     keras.layers.Dense(10, activation='softmax')
        # ])
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(self.X.shape[1], self.X.shape[2], self.X.shape[3])),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
            keras.layers.Dropout(0.25),

            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def train_model(self, batch_size=16, epochs=50):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)

        # Normalize only after splitting to prevent data leakage
        X_train -= np.mean(X_train, axis=0)
        X_train /= np.std(X_train, axis=0)

        X_test -= np.mean(X_test, axis=0)
        X_test /= np.std(X_test, axis=0)

        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)

        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        correct_preds = np.sum(y_test == y_pred_classes)
        print(f'Correct predictions: {correct_preds} out of {len(y_test)}')


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()
    data_loader = DataLoader(folder_path)
    trainer = ModelTrainer(data_loader.X, data_loader.y)
    trainer.train_model()
