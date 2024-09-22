import numpy as np

from neural_network.loss_functions.LossCommon import Loss


class LossBinaryCrossEntropy(Loss):

    def __init__(self):
        super().__init__()
        self.dinputs = None

    def forward(self, prediction, target):

        pred_clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
        sample_losses = -(target * np.log(pred_clipped) + (1 - target) * np.log(1 - pred_clipped))

        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, target):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(target / clipped_dvalues - (1 - target) / (1 - clipped_dvalues)) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples
