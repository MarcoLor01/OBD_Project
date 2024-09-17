import numpy as np

from neural_network.metrics_implementations.Accuracy import Accuracy


class Mape(Accuracy):
    def __init__(self):
        super().__init__()
        self.epsilon = None

    def initialize(self, y, recalculate=False):
        # Inizializza epsilon per evitare la divisione per zero,
        # e per gestire potenziali valori estremamente piccoli in y
        if self.epsilon is None or recalculate:
            self.epsilon = 1e-7  # Valore di default per evitare divisione per zero

    # Calcolo della MAPE come accuratezza
    def compare(self, predictions, y):
        # Calcola l'errore percentuale assoluto medio
        absolute_percentage_errors = np.abs((y - predictions) / np.maximum(np.abs(y), self.epsilon)) * 100

        return absolute_percentage_errors
