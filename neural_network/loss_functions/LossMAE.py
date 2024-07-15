import numpy as np

from neural_network.loss_functions.LossCommon import Loss


class Mae(Loss):

    def __init__(self):
        self.dinputs = None

    def forward(self, output, target_class):
        sample_losses = np.mean(np.abs(target_class - output), axis = -1)
        return sample_losses

    def backward(self, dvalues, target_class):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(target_class - dvalues) / outputs
        self.dinputs = self.dinputs / samples
