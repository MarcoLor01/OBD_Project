import matplotlib.pyplot as plt
import csv
import os

import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix


def write_on_csv(accuracy, precision, recall, f1_score, filename='metrics_results.csv'):
    file_exists = os.path.isfile(filename)

    # Open the file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(['Accuracy', 'Precision', 'Recall', 'F1 Score'])

        # Write the metrics to the file
        writer.writerow([accuracy, precision, recall, f1_score])


def plot_training_metrics(loss_history, accuracy_history=None):
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(12, 5))

    # Plot della Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Training Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    if accuracy_history:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy_history, label='Training Accuracy', marker='o')
        plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.arange(cm.shape[1]),
                yticklabels=np.arange(cm.shape[0]))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Esempio di utilizzo

# TP