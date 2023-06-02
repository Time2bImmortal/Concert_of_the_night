
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def choose_folder():
    root = tk.Tk()
    # Hide the main window
    root.withdraw()

    folder_path = filedialog.askdirectory()
    return folder_path

def build_model(input_shape):

    # create model
    model = keras.Sequential()
    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # 2nd conv layer

def prepare_datasets(test_size, validation_size):
    pass
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size)

    X_train, X_validation, X_test = X_train[..., np.newaxis], X_validation[..., np.newaxis], X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test
if __name__ == "__main__":

    #  create, train, validations sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build the cnn net
    model = build_model(input_shape)
    # train the cnn
    # Evaluate the CNN on the test set
    # Make predictions on a sample