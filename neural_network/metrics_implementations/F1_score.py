import numpy as np


class F1Score:

    def initialize(self, y):
        pass

    def __init__(self):
        self.accumulated_true_positives = None
        self.accumulated_false_positives = None
        self.accumulated_false_negatives = None

    def calculate(self, predictions, y):
        comparisons = F1Score.compare(predictions, y)
        self.accumulated_true_positives += np.sum(comparisons['tp'])
        self.accumulated_false_positives += np.sum(comparisons['fp'])
        self.accumulated_false_negatives += np.sum(comparisons['fn'])

        precision = self.accumulated_true_positives / (
                    self.accumulated_true_positives + self.accumulated_false_positives + 1e-10)
        recall = self.accumulated_true_positives / (
                    self.accumulated_true_positives + self.accumulated_false_negatives + 1e-10)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1

    def calculated_accumulated(self):
        precision = self.accumulated_true_positives / (
                    self.accumulated_true_positives + self.accumulated_false_positives + 1e-10)
        recall = self.accumulated_true_positives / (
                    self.accumulated_true_positives + self.accumulated_false_negatives + 1e-10)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1

    def new_pass(self):
        self.accumulated_true_positives = 0
        self.accumulated_false_positives = 0
        self.accumulated_false_negatives = 0

    @staticmethod
    def compare(predictions, y):
        tp = (predictions == 1) & (y == 1)
        fp = (predictions == 1) & (y == 0)
        fn = (predictions == 0) & (y == 1)

        return {
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
