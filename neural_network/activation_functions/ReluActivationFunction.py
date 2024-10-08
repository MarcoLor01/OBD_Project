import numpy as np


def predictions(outputs):
    return outputs


class Relu:

    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, input_units, training):
        self.inputs = input_units
        self.output = np.maximum(0, input_units)

    def __str__(self):
        return f"Attivazione Relu"

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
