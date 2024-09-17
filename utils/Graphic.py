import matplotlib.pyplot as plt
import numpy as np


# Function for printing accuracy and loss


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

    # Plot dell'accuratezza se disponibile
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
    # Creare il grafico
    plt.figure(figsize=(8, 6))

    # Scatter plot: labels sull'asse x, predictions sull'asse y

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

    # Aggiungere una legenda
    plt.legend()

    # Mostrare il grafico
    plt.show()


def plot_residuals(model, predictions, y_test):
    """
    Calcola e visualizza l'istogramma dei residui per un modello di regressione.

    Parameters:
    - model: Il modello di regressione già addestrato
    - X_test: Dati di input di test
    - y_test: Valori osservati (target reali) di test
    """
    # Ottieni le predizioni del modello
    # Calcola i residui (errore: valore osservato - valore predetto)
    residuals = y_test - predictions

    mean_error = np.mean(residuals)
    median_error = np.median(residuals)
    # Crea l'istogramma dei residui
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)

    # Imposta i titoli e le etichette
    plt.title('Istogramma dei Residui', fontsize=16)
    plt.xlabel('Residui', fontsize=14)
    plt.ylabel('Frequenza', fontsize=14)

    # Mostra il grafico
    plt.grid(True)
    plt.show()

    print("Errore medio commesso sul test set: ", mean_error)
    print("Errore mediano commesso sul test set: ", median_error)
