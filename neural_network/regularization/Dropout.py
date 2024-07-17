import numpy as np


class Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate
        self.inputs = None
        self.mask = None
        self.output = None
        self.dinputs = None

    def __str__(self):
        return f"Dropout di rate: ({1 - self.rate})"

    def forward(self, inputs, training):
        self.inputs = inputs
        self.mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = self.inputs * self.mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask
        