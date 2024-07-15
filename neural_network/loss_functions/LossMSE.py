import numpy as np

from neural_network.loss_functions.LossCommon import Loss


class Mse(Loss):

    def __init__(self):
        self.dinputs = None

    def forward(self, output, target_class):
        sample_losses = np.mean((target_class - output)**2, axis = -1)
        return sample_losses

    def backward(self, dvalues, target_class):
        samples = len(dvalues)
        output = len(dvalues[0])
        self.dinputs = -2 * (target_class - dvalues) / output
        self.dinputs = self.dinputs / samples
