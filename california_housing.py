import argparse

from sklearn.model_selection import train_test_split

from CrossValidation import validation, print_best_model
from dataset.california_housing_regression.preprocess_california import california_housing
from neural_network.DenseLayer import DenseLayer
from neural_network.Model import Model
from neural_network.activation_functions.LinearActivation import ActivationLinear
from neural_network.activation_functions.TanhActivationFunction import Tanh
from neural_network.loss_functions.RMSE import Rmse
from neural_network.optimizers.Adam import Adam
from neural_network.regularization.EarlyStopping import EarlyStopping
from utils.Graphic import graphic_regression_difference, plot_residuals
from utils.UtilsFunctions import shuffle_data


def train_and_validate(X_train, y_train, n_output=1, number_of_folders=5, epochs=30, multithread=True):
    """
    Esegue la validazione sul training set usando la cross-validation.

    :param X_train: Set di dati di addestramento
    :param y_train: Etichette del set di dati di addestramento
    :param n_output: Numero di neuroni di uscita del modello (es. 2 per classificazione binaria)
    :param number_of_folders: Numero di fold per la cross-validation
    :param epochs: Numero di epoche per l'addestramento
    :param multithread: Abilitare o meno il multithreading
    :return: Modello ottimale dopo la cross-validation
    """
    # Parametri da impostare per la chiamata alla Cross-Validation

    layer_combination = [[64, 32], [32, 16], [16, 8]]
    regularizers = ["l2", "l1", None]
    optimizers = ["adam", "rmsprop", "sgd_momentum"]
    dropout = [True, False]
    activation_functions = ["relu", "tanh"]

    best_model = validation(X_train, y_train,
                            n_output=n_output, number_of_folders=number_of_folders,
                            epochs=epochs, layer_combination=layer_combination,
                            regularizers=regularizers, optimizers=optimizers, dropout=dropout,
                            activation_functions=activation_functions, multithread=multithread)
    print_best_model(best_model['model'])


def retrain_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Riaddestra il miglior modello sull'intero training set e lo valuta sul test set.

    :param X_train: Set di dati di addestramento
    :param X_test: Set di dati di test
    :param y_train: Etichette del set di dati di addestramento
    :param y_test: Etichette del set di dati di test
    :return: Statistiche delle performance del modello (accuracy, precision, recall, f1_score)
    """

    # ===== STRUTTURA DELLA MIGLIOR RETE NEURALE =====

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


    loss_history, _, _, _ = best_model_retrained.train(X_train, y_train, val_data=(X_val, y_val), epochs=100,
                                                       batch_size=64, print_every=100, history=True,
                                                       early_stopping_metric="valid_loss", task_type="regression")

    y_pred = best_model_retrained.predict(X_test)
    loss = best_model_retrained.loss.calculate(y_pred, y_test)

    print("\n========== Stats of the Best Model on the Test Set ==========\n ")
    print("Loss: ", loss)

    graphic_regression_difference(y_test, y_pred)
    plot_residuals(y_pred, y_test)

    return


def main():
    X_train, y_train, X_test, y_test = california_housing()
    X_train, y_train = shuffle_data(X_train, y_train)
    X_test, y_test = shuffle_data(X_test, y_test)

    parser = argparse.ArgumentParser(description="Addestramento e valutazione di una rete neurale.")
    parser.add_argument("mode", choices=["crossvalidation", "test"],
                        help="Seleziona 'crossvalidation' per eseguire la cross-validation o 'test' per eseguire il "
                             "test sul test set.")
    args = parser.parse_args()

    if args.mode == "crossvalidation":
        train_and_validate(X_train, y_train)
        print("\nCross-validation completata, miglior modello trovato.")

    elif args.mode == "test":
        retrain_and_evaluate(X_train, X_test, y_train, y_test)
        print("\nTest completato.")


if __name__ == "__main__":
    main()
