from CrossValidation import validation
from dataset.california_housing_regression.preprocess_california import california_housing
from neural_network.DenseLayer import DenseLayer
from neural_network.Model import Model
from neural_network.activation_functions.LinearActivation import ActivationLinear
from neural_network.activation_functions.ReluActivationFunction import Relu
from neural_network.loss_functions.RMSE import Rmse
from neural_network.optimizers.Adam import Adam
from neural_network.regularization.EarlyStopping import EarlyStopping
from utils.Graphic import graphic_regression_difference, plot_residuals

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = california_housing()

    best_model = validation(X_train, y_train, n_output=1, number_of_folders=5, epochs=30,
                            multithread=True)

    # graphic_regression_difference(y_test, y_pred)
    # plot_residuals(y_pred, y_test)
