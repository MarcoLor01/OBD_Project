import numpy as np


def compare_test_multiclass(predictions, y):
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    if y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)

    num_classes = len(np.unique(y))

    accuracy = precision = recall = f1_score = 0

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for cls in range(num_classes):
        tp = ((predictions == cls) & (y == cls)).sum()
        fp = ((predictions == cls) & (y != cls)).sum()
        fn = ((predictions != cls) & (y == cls)).sum()
        tn = ((predictions != cls) & (y != cls)).sum()

        precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_scores.append(precision_cls)

        recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_scores.append(recall_cls)

        # F1 Score
        if precision_cls + recall_cls > 0:
            f1_score_cls = 2 * (precision_cls * recall_cls) / (precision_cls + recall_cls)
        else:
            f1_score_cls = 0
        f1_scores.append(f1_score_cls)

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    f1_score = np.mean(f1_scores)

    return accuracy, precision, recall, f1_score


def compare_test(predictions, y):
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    tp = ((predictions == 1) & (y == 1)).sum()

    fp = ((predictions == 1) & (y == 0)).sum()

    fn = ((predictions == 0) & (y == 1)).sum()

    tn = ((predictions == 0) & (y == 0)).sum()

    # Precisione: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Richiamo (Recall): TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Accuratezza: (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    # F1 Score: 2 * (precision * recall) / (precision + recall)
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return accuracy, precision, recall, f1_score


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
