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
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
