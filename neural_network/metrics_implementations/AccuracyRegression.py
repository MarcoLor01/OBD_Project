import numpy as np

from neural_network.metrics_implementations.Accuracy import Accuracy


class AccuracyRegression(Accuracy):
    def __init__(self):
        self.accuracy = None

    def initialize(self, y, recalculate=False):
        if self.accuracy is None or recalculate:
            self.accuracy = np.std(y) / 250

    def compare(self, predictions, target_class):
        return np.absolute(predictions - target_class) < self.accuracy

