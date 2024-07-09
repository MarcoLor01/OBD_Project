import numpy as np
from nnfs.datasets import spiral_data

from neural_network.activation_functions.ReluActivationFunction import Relu
from neural_network.DenseLayer import DenseLayer
from neural_network.loss_functions.LossCommon import regularization_loss
from neural_network.loss_functions.SoftmaxCrossEntropy import SoftmaxCrossEntropy
from neural_network.metrics_implementations.Metrics import accuracy_metric
from neural_network.optimizers.Adam import Adam

X, y = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

dense1 = DenseLayer(2, 128, l1_regularization_weights=5e-4, l1_regularization_bias=5e-4)
activation1 = Relu()
dense2 = DenseLayer(128, 64, l1_regularization_weights=5e-4, l1_regularization_bias=5e-4)
activation2 = Relu()
dense3 = DenseLayer(64, 3)

loss = SoftmaxCrossEntropy()
optimizer = Adam(learning_rate=0.05, decay=5e-7)

loss_history = []
accuracy_history = []

for epoch in range(0, 10001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    reg_loss = regularization_loss(dense1) + regularization_loss(dense2) + regularization_loss(dense3)
    loss_result = loss.forward(dense3.output, y) + reg_loss

    predictions = np.argmax(loss.output, axis=1)
    accuracy = accuracy_metric(predictions, y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss_result:.3f}, ' +
              f'actual learning rate: {optimizer.current_learning_rate:.4f}'
              )
        loss_history.append(loss_result)
        accuracy_history.append(accuracy)

    loss.backward(loss.output, y)
    dense3.backward(loss.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.decay_learning_rate_step()
    optimizer.update_weights(dense1)
    optimizer.update_weights(dense2)
    optimizer.update_weights(dense3)
    optimizer.post_step_learning_rate()

