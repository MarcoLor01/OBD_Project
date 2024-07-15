import numpy as np

from neural_network.metrics_implementations.Accuracy import Accuracy


class AccuracyCategorical(Accuracy):

    def initialize(self, y):
        pass

    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
