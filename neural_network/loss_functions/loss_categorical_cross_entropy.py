import numpy as np

from neural_network.loss_functions.loss_common import loss


class loss_categorical_cross_entropy(loss):

    def __init__(self):
        self.dinputs = None

    def forward(self, output, target_class):
        samples = len(output)  # Length of predictions
        y_prediction_clipped = np.clip(output, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(target_class.shape) == 1:
            correct_confidences = y_prediction_clipped[
                range(samples),
                target_class
            ]
        # Mask values - only for one-hot encoded labels
        elif len(target_class.shape) == 2:
            correct_confidences = np.sum(
                y_prediction_clipped * target_class,
                axis=1
            )
        else:
            raise ValueError("target_class must be a 1D or 2D array")

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, target_class):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(target_class.shape) == 1:
            target_class = np.eye(labels)[target_class]

        self.dinputs = -target_class / dvalues
        self.dinputs = self.dinputs / samples  # Normalization of Gradient
