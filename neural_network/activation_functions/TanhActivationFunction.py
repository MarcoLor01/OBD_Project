import numpy as np


class Tanh:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, input_units, training):
        self.inputs = input_units
        self.output = np.tanh(input_units)  # Funzione di attivazione tanh

    def __str__(self):
        return "Attivazione Tanh"

    def backward(self, dvalues):
        # Derivata di tanh è 1 - tanh^2
        self.dinputs = dvalues * (1 - np.power(self.output, 2))