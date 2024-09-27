import numpy as np


class Softmax:

    def __init__(self):
        self.dinputs = None
        self.output = None

    def __str__(self):
        return f"Attivazione Softmax"

    def forward(self, input_units, training):
        exp_values = np.exp(input_units - np.max(input_units, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=-1)
