import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import os
import gzip
import random
from tkinter import filedialog
import tkinter as tk

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    with gzip.open(data_path, 'rt') as fp:  # 'rt' mode to open as text file
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return  X, y

def load_data_from_folder(folder_path, num_files_per_treatment=50):
    """Loads training dataset from a folder with multiple gzip files.

        :param folder_path (str): Path to the folder containing gzip files
        :param num_files_per_treatment (int): Number of files to load per light treatment
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    X, y = [], []
    correct_shape = None
    # Assume light treatments are folders under the main folder
    for treatment in os.listdir(folder_path):
        treatment_path = os.path.join(folder_path, treatment)
        if os.path.isdir(treatment_path):
            all_files = [f for f in os.listdir(treatment_path) if f.endswith('.gz')]
            selected_files = random.sample(all_files, num_files_per_treatment)  # randomly select files
            for file in selected_files:
                X_temp, y_temp = load_data(os.path.join(treatment_path, file))
                if correct_shape is None:
                    correct_shape = X_temp.shape
                if X_temp.shape == correct_shape:
                    X.append(X_temp)
                    y.append(y_temp)

    if X:  # check if X is not empty
        X = np.concatenate(X, axis=0)  # concatenate along the first axis
        y = np.concatenate(y, axis=0)
        print("Data succesfully loaded!")
    else:
        print("No valid data loaded!")

    return X, y


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # open file dialog to choose a folder
    folder_path = filedialog.askdirectory()

    # load data
    X, y = load_data_from_folder(folder_path)
    #Normalize
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    # X = X[..., np.newaxis]
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network topology
    # model = keras.Sequential([
    #
    #     # input layer
    #     keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),
    #
    #     # 1st dense layer
    #     keras.layers.Dense(512, activation='relu'),
    #
    #     # 2nd dense layer
    #     keras.layers.Dense(256, activation='relu'),
    #
    #     # 3rd dense layer
    #     keras.layers.Dense(64, activation='relu'),
    #
    #     # output layer
    #     keras.layers.Dense(10, activation='softmax')
    # ])
    # Create CNN model
    # model = keras.Sequential([
    #     # Input layer, reshape to 2D image format
    #     keras.layers.Reshape((X.shape[1], X.shape[2], 1), input_shape=(X.shape[1], X.shape[2])),
    #
    #     # 1st conv layer
    #     keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    #     keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    #     keras.layers.Dropout(0.04),  # Dropout layer
    #
    #     # 2nd conv layer
    #     keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    #     keras.layers.Dropout(0.06),  # Dropout layer
    #
    #     # 3rd conv layer
    #     keras.layers.Conv2D(128, (2, 2), activation='relu'),
    #     keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
    #     keras.layers.Dropout(0.16),  # Dropout layer
    #
    #     # Flatten the tensor output from the previous layer
    #     keras.layers.Flatten(),
    #
    #     # Dense layers (aka. fully connected layers)
    #     keras.layers.Dense(64, activation='relu'),
    #     keras.layers.Dropout(0.1),  # Dropout layer
    #     keras.layers.Dense(32, activation='relu'),
    #     keras.layers.Dropout(0.1),  # Dropout layer
    #
    #     # Output layer
    #     keras.layers.Dense(4, activation='softmax')  # assuming you have 4 light treatments hence 4 classes
    # ])
    model = keras.Sequential([
        # Input layer. 'None' in the second place (for timesteps) allows input sequences of variable length.
        keras.layers.Input(shape=(None, X.shape[2])),

        # 1st RNN layer
        keras.layers.SimpleRNN(32, return_sequences=True),
        keras.layers.Dropout(0.2),

        # 2nd RNN layer
        keras.layers.SimpleRNN(64, return_sequences=True),
        keras.layers.Dropout(0.2),

        # 3rd RNN layer
        keras.layers.SimpleRNN(128),
        keras.layers.Dropout(0.2),

        # Dense layer
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.1),

        # Output layer
        keras.layers.Dense(4, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)
    # load test data
    X_test, y_test = load_data_from_folder(folder_path)

    # make predictions on the test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # calculate performance
    correct_preds = np.sum(y_test == y_pred_classes)
    print(f'Correct predictions: {correct_preds} out of {len(y_test)}')
