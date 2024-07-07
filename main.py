from nnfs.datasets import spiral_data
import numpy as np
from neural_network.activation_functions.relu_act_function import relu
from neural_network.dense_layer import dense_layer
from neural_network.metrics_implementations import metrics
from neural_network.loss_functions import loss_categorical_cross_entropy

X, y = spiral_data(samples=100, classes=3)

dense1 = dense_layer(2, 64)
activation1 = relu()
dense2 = dense_layer(64, 3)
activation2 = loss_categorical_cross_entropy()


