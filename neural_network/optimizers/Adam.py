import numpy as np


class Adam:  # Adaptive gradient

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.current_learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def decay_learning_rate_step(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iteration)

    def update_weights(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.dweights = layer.dweights.astype(float)
        layer.dbiases = layer.dbiases.astype(float)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iteration + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iteration + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (
                1 - self.beta_2) * layer.dweights ** 2  # PROBLEMA: LAYER.DWEIGHTS = OBJECT
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iteration + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iteration + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (
                np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (
                np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_step_learning_rate(self):
        self.iteration += 1

    def __str__(self):
        if self.decay != 0.:
            return f"Ottimizzatore utilizzato: Adam con learning rate: {self.learning_rate} e tasso di decadimento: {self.decay}"
        else:
            return f"Ottimizzatore utilizzato: Adam con learning rate: {self.learning_rate}"
