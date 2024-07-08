class sgd:

    def __init__(self, learning_rate=1.0, decay=0.):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.current_learning_rate = 0

    def decay_learning_rate_step(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))

    def update_weights(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

    def post_step_learning_rate(self):
        self.iteration += 1
