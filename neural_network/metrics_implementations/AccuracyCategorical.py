import numpy as np

from neural_network.metrics_implementations.Accuracy import Accuracy


class AccuracyCategorical(Accuracy):
    def __init__(self, *, binary=False):
        super().__init__()
        self.binary = binary

    # No initialization is needed
    def initialize(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary:
            # Se predictions è un'uscita softmax (probabilità), prendi l'argmax
            if predictions.ndim > 1:
                predictions = np.argmax(predictions, axis=1)

            # Se y è one-hot encoded, prendi l'argmax anche di y
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
        else:
            # In caso binario, si presume che predictions e y siano già valori binari (0 o 1)
            predictions = (predictions > 0.5).astype(int)

        return predictions == y
