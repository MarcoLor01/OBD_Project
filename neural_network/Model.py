import pickle

import numpy as np

from neural_network.FirstLayer import FirstLayer
from neural_network.activation_functions.SoftmaxActivationFunction import Softmax
from neural_network.loss_functions.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy
from neural_network.loss_functions.SoftmaxCrossEntropy import SoftmaxCrossEntropy
import copy


class Model:
    def __init__(self):
        self.early_stopping = None
        self.accuracy = None
        self.output_activation = None
        self.input_layer = None
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.trainable = []
        self.softmax_output = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy, early_stopping=None):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.early_stopping = early_stopping

    def train(self, X, y, *, epochs=1, print_every=1, val_data=None, batch_size=None):
        validation_early_stop = -1
        X_val = None
        y_val = None
        self.accuracy.initialize(y)

        train_steps = 1

        if val_data is not None:
            X_val, y_val = val_data

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
            if val_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    start = step * batch_size
                    end = start + batch_size
                    batch_X = X[start:end]
                    batch_y = y[start:end]

                output = self.forward(batch_X, training=True)
                data_loss, reg_loss = self.loss.calculate(output, batch_y, include_reg=True)
                loss = data_loss + reg_loss
                prediction = self.output_activation.predictions(output)
                accuracy = self.accuracy.calculate(prediction, batch_y)

                self.backward(output, batch_y)

                self.optimizer.decay_learning_rate_step()
                for layer in self.trainable:
                    self.optimizer.update_weights(layer)
                self.optimizer.post_step_learning_rate()

                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {reg_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculated_accumulated(include_reg=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculated_accumulated()
            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            if val_data is not None:
                validation_early_stop = self.evaluate(X_val, y_val, batch_size=batch_size)
            if self.early_stopping is not None and validation_early_stop != -1:
                if self.early_stopping(validation_early_stop):
                    print(f'Early stopping at epoch {epoch}')
                    return

    def finalize(self):
        self.input_layer = FirstLayer()
        layer_count = len(self.layers)
        self.trainable = []
        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable.append(self.layers[i])

        self.loss.set_trainable(self.trainable)

        if isinstance(self.layers[-1], Softmax) and isinstance(self.output_activation, LossCategoricalCrossEntropy):
            self.softmax_output = SoftmaxCrossEntropy()

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        layer = None

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        if layer is not None:
            return layer.output

    def backward(self, output, y):
        if self.softmax_output is not None:
            self.softmax_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                start = step * batch_size
                end = start + batch_size
                batch_X = X_val[start:end]
                batch_y = y_val[start:end]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            prediction = self.output_activation.predictions(output)
            self.accuracy.calculate(prediction, batch_y)

        validation_loss = self.loss.calculated_accumulated()
        validation_accuracy = self.accuracy.calculated_accumulated()

        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')
        return validation_loss

    def predict(self, X, *, batch_size=None):
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []

        for step in range(prediction_steps):
            if batch_size is not None:
                start = step * batch_size
                end = None if batch_size is None else start + batch_size
                batch_X = X[start:end]
            else:
                batch_X = X

            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)


        return np.concatenate(output, axis=0)

    def save(self, path):

        model = copy.deepcopy(self)

        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for properties in ['inputs', 'output', 'dinputs',
                               'dweights', 'dbiases']:
                layer.__dict__.pop(properties, None)
        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
