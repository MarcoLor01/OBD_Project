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

dense1 = DenseLayer(2, 64)
activation1 = Relu()
dense2 = DenseLayer(64, 3)
loss = SoftmaxCrossEntropy()
optimizer = Adam(learning_rate=0.05, decay=5e-7)

loss_history = []
accuracy_history = []

for epoch in range(0, 10001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    reg_loss = regularization_loss(dense1) + regularization_loss(dense2)
    loss_result = loss.forward(dense2.output, y) + reg_loss

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
    dense2.backward(loss.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.decay_learning_rate_step()
    optimizer.update_weights(dense1)
    optimizer.update_weights(dense2)
    optimizer.post_step_learning_rate()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss_result = loss.forward(dense2.output, y_test)

predictions = np.argmax(loss.output, axis=1)
accuracy = accuracy_metric(predictions, y_test)
print(f'Test acc: {accuracy:.3f}, ' + f'Test loss: {loss_result:.3f}')
