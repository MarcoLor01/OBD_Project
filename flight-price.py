from dataset.fashion_mnist_classification.preprocess_fashion_mnist import shuffle_data
from dataset.flight_price_prediction.preprocess_flight_price import flight_price_dataset
from neural_network.DenseLayer import DenseLayer
from neural_network.Model import Model
from neural_network.RandomizedSearch import print_model
from neural_network.activation_functions.LinearActivation import ActivationLinear
from neural_network.activation_functions.ReluActivationFunction import Relu
from neural_network.loss_functions.RMSE import Rmse
from neural_network.metrics_implementations.Mape import Mape
from neural_network.optimizers.RmsProp import Rmsprop
from neural_network.regularization.EarlyStopping import EarlyStopping
from utils.Graphic import plot_training_metrics, plot_residuals, graphic_regression_difference

X_train, y_train, X_valid, y_valid, X_test, y_test, X_scaler, y_scaler = flight_price_dataset()

X_train, y_train = shuffle_data(X_train, y_train)
X_valid, y_valid = shuffle_data(X_valid, y_valid)
X_test, y_test = shuffle_data(X_test, y_test)

n_inputs = X_train.shape[1]

n_output = 1
n_neurons_first_hidden_layer = int(2 / 3 * n_inputs + n_output)
n_neurons_second_hidden_layer = int(n_neurons_first_hidden_layer / 2)

model = Model()
model.add_layer(DenseLayer(X_train.shape[1], 32, initialization="He", l2_regularization_weights=0.001, l2_regularization_bias=0.001))
model.add_layer(Relu())
model.add_layer(DenseLayer(32, 64, initialization="He", l2_regularization_weights=0.001, l2_regularization_bias=0.001))
model.add_layer(Relu())
model.add_layer(DenseLayer(64, 128, initialization="He", l2_regularization_weights=0.001, l2_regularization_bias=0.001))
model.add_layer(Relu())
model.add_layer(DenseLayer(128, n_output, initialization="He", l2_regularization_weights=0.001, l2_regularization_bias=0.001))
model.add_layer(ActivationLinear())

model.set(
    loss=Rmse(),
    optimizer=Rmsprop(learning_rate=0.01, decay=5e-5),
    accuracy=Mape(),
    early_stopping=EarlyStopping(patience=10, min_delta=0.1)
)

model.finalize()
print_model(-1, model)
loss_history, accuracy_history, val_loss_history, val_accuracy_history = model.train(X_train, y_train, val_data=(X_valid, y_valid), epochs=40, batch_size=128, print_every=100)

confidences = model.predict(X_test)
original_predictions = y_scaler.inverse_transform(confidences)
original_y_test = y_scaler.inverse_transform(y_test)
mse = Rmse()
loss = mse.forward(original_predictions, original_predictions)

plot_residuals(model, X_test, original_y_test)
graphic_regression_difference(original_y_test, original_predictions)
plot_training_metrics(loss_history, accuracy_history)
plot_training_metrics(val_loss_history, val_accuracy_history)
