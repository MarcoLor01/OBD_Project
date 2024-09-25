import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix


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


def graphic_regression_difference(labels, predictions):
    plt.figure(figsize=(8, 6))

    plt.scatter(labels, predictions, color='blue', label='Predizioni vs Etichette')
    for i in range(len(labels)):
        if labels[i] > 50000:
            print("Label asse x:", labels[i], "Asse y: ", predictions[i])
    max_val = max(max(labels), max(predictions))  # valore massimo tra labels e predictions
    min_val = min(min(labels), min(predictions))  # valore minimo tra labels e predictions
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Linea di Perfetta Accuratezza')

    plt.title('Confronto tra Predizioni e Etichette')
    plt.xlabel('Etichette (Labels)')
    plt.ylabel('Predizioni')

    plt.legend()

    plt.show()


def plot_residuals(predictions, y_test):
    residuals = y_test - predictions

    mean_error = np.mean(residuals)
    median_error = np.median(residuals)

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)

    plt.title('Istogramma dei Residui', fontsize=16)
    plt.xlabel('Residui', fontsize=14)
    plt.ylabel('Frequenza', fontsize=14)

    plt.grid(True)
    plt.show()

    print("Errore medio commesso sul test set: ", mean_error)
    print("Errore mediano commesso sul test set: ", median_error)


def print_confusion_matrix(y_true, y_pred, class_labels=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))

    if class_labels is None:
        class_labels = np.arange(cm.shape[1])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def print_fashion_mnist_stats(X_train, y_train):

    plt.figure(figsize=(8, 6))
    plt.hist(y_train, bins=np.arange(11) - 0.5, edgecolor='black', alpha=0.7)
    plt.xticks(np.arange(10))
    plt.title("Distribuzione delle Classi nel Dataset Fashion MNIST", fontsize=16)
    plt.xlabel("Classe", fontsize=14)
    plt.ylabel("Frequenza", fontsize=14)
    plt.grid(True)
    plt.show()

    class_labels = ['T-shirt/Top', 'Pantaloni', 'Pullover', 'Vestito', 'Cappotto',
                    'Sandalo', 'Camicia', 'Sneaker', 'Borsa', 'Stivale']

    num_examples = 6
    plt.figure(figsize=(12, 8))

    for i in range(num_examples):
        plt.subplot(2, 3, i + 1)
        img = X_train[i].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f"Classe: {class_labels[y_train[i]]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()