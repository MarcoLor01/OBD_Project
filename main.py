from nnfs.datasets import spiral_data
import numpy as np
from neural_network.activation_functions.relu_act_function import relu
from neural_network.activation_functions.softmax_act_function import softmax
from neural_network.dense_layer import dense_layer
from neural_network.loss_functions.loss_categorical_cross_entropy import loss_categorical_cross_entropy
from neural_network.metrics_implementations import metrics


X, y = spiral_data(samples=100, classes=3)

dense1 = dense_layer(2, 3)
activation1 = relu()
dense2 = dense_layer(3, 3)
activation2 = softmax()
loss_function = loss_categorical_cross_entropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y)
# Print loss value
print('loss:', loss)

predictions = np.argmax(activation2.output, axis=1)

accuracy_value = metrics.accuracy_metric(predictions, y)
print('acc:', accuracy_value)

