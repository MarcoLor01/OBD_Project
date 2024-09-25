import numpy as np
from neural_network.loss_functions.LossCommon import Loss


class LossCategoricalCrossEntropy(Loss):

    def __init__(self):
        super().__init__()
        self.dinputs = None

    def forward(self, output, target_class):
        samples = len(output)

        y_prediction_clipped = np.clip(output, 1e-7, 1 - 1e-7)

        if len(target_class.shape) == 1:
            correct_confidences = y_prediction_clipped[
                range(samples),
                target_class
            ]

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

        if np.isnan(self.dinputs).any():
            print(self.dinputs.shape)
            print(self.dinputs)
            print(dvalues)
            print(target_class)
            print("Warning: NaN detected in dinputs!")

        self.dinputs = self.dinputs / samples
