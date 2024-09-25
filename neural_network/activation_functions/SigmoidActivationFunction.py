import numpy as np


def predictions(outputs):
    return (outputs > 0.5) * 1


class Sigmoid:

    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def __str__(self):
        return f"Attivazione Sigmoid"

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
