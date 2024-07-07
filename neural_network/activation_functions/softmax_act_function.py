import numpy as np


# Implementation of softmax activation function

class softmax:

    def __init__(self):
        self.output = None

    def forward(self, input_units):
        # Get unnormalized probabilities
        exp_values = np.exp(input_units - np.max(input_units, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
