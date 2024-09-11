import numpy as np

from neural_network.loss_functions.LossCommon import Loss


class Mse(Loss):

    def __init__(self):
        super().__init__()
        self.dinputs = None

    def forward(self, output, target_class):

        assert output.shape == target_class.shape, f"Shapes non corrispondenti: output {output.shape}, target {target_class.shape}"

        # Calcolo della MSE
        sample_losses = np.mean((target_class - output) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues, target_class):

        assert dvalues.shape == target_class.shape, f"Shapes non corrispondenti: dvalues {dvalues.shape}, target {target_class.shape}"

        samples = len(dvalues)
        # Numero di features nel layer precedente
        output = dvalues.shape[1]

        # Gradienti della MSE
        self.dinputs = -2 * (target_class - dvalues) / output
        self.dinputs = self.dinputs / samples
