import numpy as np


class Rmsprop:  # Adaptive gradient

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.current_learning_rate = learning_rate
        self.epsilon = epsilon
        self.rho = rho

    def decay_learning_rate_step(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iteration)

    def update_weights(self, layer):
        if not hasattr(layer, "rmsprop_weights"):
            layer.rmsprop_weights = np.zeros_like(layer.weights)
            layer.rmsprop_biases = np.zeros_like(layer.biases)

        layer.rmsprop_weights = (self.rho * layer.rmsprop_weights) + ((1-self.rho) * layer.dweights**2)
        layer.rmsprop_biases = (self.rho * layer.rmsprop_biases) + ((1-self.rho) * layer.dbiases**2)

        layer.weights += -self.current_learning_rate * (layer.dweights / (np.sqrt(layer.rmsprop_weights) + self.epsilon))
        layer.biases += -self.current_learning_rate * (layer.dbiases / (np.sqrt(layer.rmsprop_biases) + self.epsilon))

    def post_step_learning_rate(self):
        self.iteration += 1


    def __str__(self):
        if self.decay != 0.:
            return f"Ottimizzatore utilizzato: RmsProp con learning rate: {self.learning_rate} e tasso di decadimento: {self.decay}"
        else:
            return f"Ottimizzatore utilizzato: RmsProp con learning rate: {self.learning_rate}"
