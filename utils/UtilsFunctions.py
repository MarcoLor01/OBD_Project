from matplotlib import pyplot as plt
from sklearn.utils import shuffle

import numpy as np
import pandas as pd

BATCH_SIZE = 128


def convert_to_numeric_array(y):
    y_array = np.array(y)

    if y_array.ndim > 1:
        y_array = y_array.ravel()

    y_numeric = pd.to_numeric(y_array, errors='coerce')
    y_numeric = np.nan_to_num(y_numeric, nan=0)
    y_numeric = y_numeric.astype(int)

    return y_numeric


def total_step(X):
    steps = X.shape[0] // BATCH_SIZE
    if steps * BATCH_SIZE < X.shape[0]:
        steps += 1
    return steps


def shuffle_data(X_train, y_train):
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    return X_train, y_train


def print_stats(X_train, y_train):
    plt.figure(figsize=(8, 6))
    plt.hist(y_train, bins=np.arange(11) - 0.5, edgecolor='k', alpha=0.7)
    plt.xticks(np.arange(10))
    plt.title('Distribuzione delle Etichette nel Dataset MNIST', fontsize=16)
    plt.xlabel('Cifre', fontsize=14)
    plt.ylabel('Frequenza', fontsize=14)
    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle('Esempi di Immagini del Dataset MNIST', fontsize=16)
    for i, ax in enumerate(axes.flat):
        img = X_train[i].reshape(28, 28)  # Reshape a 28x28 pixel
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Etichetta: {y_train[i]}')
        ax.axis('off')
    plt.show()


def compare_test(predictions, y):

    assert predictions.shape[0] == y.shape[0], "Le dimensioni di predictions e y devono essere uguali"

    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    tp = np.sum((predictions == 1) & (y == 1))
    fp = np.sum((predictions == 1) & (y == 0))
    fn = np.sum((predictions == 0) & (y == 1))
    tn = np.sum((predictions == 0) & (y == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score
