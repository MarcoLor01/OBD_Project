import numpy as np
from sklearn.model_selection import train_test_split

from CrossValidation import validation, print_best_model
from dataset.california_housing_regression.preprocess_california import california_housing
from neural_network.DenseLayer import DenseLayer
from neural_network.Model import Model
from neural_network.activation_functions.LinearActivation import ActivationLinear
from neural_network.activation_functions.ReluActivationFunction import Relu
from neural_network.activation_functions.TanhActivationFunction import Tanh
from neural_network.loss_functions.RMSE import Rmse
from neural_network.optimizers.Adam import Adam
from neural_network.regularization.EarlyStopping import EarlyStopping
from utils.Graphic import graphic_regression_difference, plot_residuals

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = california_housing()

    # best_model = validation(X_train, y_train, n_output=1, number_of_folders=5, epochs=30,
    #                        multithread=True)
    # print_best_model(best_model['model'])

    best_model_retrained = Model()
    best_model_retrained.add_layer(DenseLayer(X_train.shape[1], 64, initialization="Glorot"))
    best_model_retrained.add_layer(Tanh())
    best_model_retrained.add_layer(DenseLayer(64, 32, initialization="Glorot"))
    best_model_retrained.add_layer(Tanh())
    best_model_retrained.add_layer(DenseLayer(32, 1, initialization="Glorot"))
    best_model_retrained.add_layer(ActivationLinear())

    best_model_retrained.set(loss=Rmse(),
                             optimizer=Adam(learning_rate=0.01, decay=0.0001),
                             accuracy=None,
                             early_stopping=EarlyStopping(patience=10, min_delta=0.001, mode="min"))

    best_model_retrained.finalize()

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    loss_history, _, _, _ = best_model_retrained.train(X_train, y_train, val_data=(X_val, y_val), epochs=200,
                                                       batch_size=64, print_every=100, history=True,
                                                       early_stopping_metric="valid_loss", task_type="regression")

    y_pred = best_model_retrained.predict(X_test)
    loss = best_model_retrained.loss.calculate(y_pred, y_test)

    print("\n========== Stats of the Best Model on the Test Set ==========\n ")
    print("Loss: ", loss)

    graphic_regression_difference(y_test, y_pred)
    plot_residuals(y_pred, y_test)
