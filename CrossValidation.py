import threading
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from neural_network.DenseLayer import DenseLayer
from neural_network.Model import Model
from neural_network.activation_functions.LinearActivation import ActivationLinear
from neural_network.activation_functions.ReluActivationFunction import Relu
from neural_network.activation_functions.SoftmaxActivationFunction import Softmax
from neural_network.activation_functions.TanhActivationFunction import Tanh
from neural_network.loss_functions.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy
from neural_network.loss_functions.RMSE import Rmse
from neural_network.metrics_implementations.AccuracyCategorical import AccuracyCategorical
from neural_network.metrics_implementations.F1_score import compare_test_multiclass
from neural_network.optimizers.Adagrad import Adagrad
from neural_network.optimizers.RmsProp import Rmsprop
from neural_network.optimizers.Sgd import Sgd
from neural_network.optimizers.Adam import Adam
from neural_network.regularization.Dropout import Dropout
from neural_network.regularization.EarlyStopping import EarlyStopping

from colorama import Fore, Style, init

init(autoreset=True)

best_model = {
    'model': Model,
    'f1_metric': 0.0,
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'loss': 9999999
}


def print_best_model(model):
    print(f"\n{Fore.CYAN}===== STRUTTURA DELLA RETE NEURALE =====\n")

    model_opt = model.optimizer  # Se vuoi stampare anche l'optimizer
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__

        if hasattr(layer, 'weights'):
            layer_description = f"{str(layer)}"
        elif hasattr(layer, 'rate'):
            layer_description = f"{str(layer)}"
        else:
            layer_description = f"Attivazione {str(layer_type)}"

        print(f"{Fore.CYAN}{i}: {layer_description}")
        print(f"{Fore.CYAN + '-' * 40}")

    # Se desideri stampare l'optimizer
    print(f"\n{Fore.CYAN} Ottimizzatore: {str(model_opt)}")

    print(Style.RESET_ALL)


def print_model(iteration, model, combination, number_of_combinations):
    if iteration >= 0:
        print(f"\n{Fore.BLUE + '=' * 40}")
        print(f"{Fore.BLUE}Combinazione Numero: {iteration + 1}")
        print(f"{Fore.BLUE + '=' * 40}\n")
    print(f"{Fore.CYAN}\n Modello {combination}/{number_of_combinations}")
    print(f"{Fore.CYAN}\n ===== STRUTTURA DELLA RETE NEURALE =====\n")
    model_opt = model.optimizer
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        if hasattr(layer, 'weights'):
            dimensions = layer.weights.shape
            layer_description = f"{str(layer)}"
        elif hasattr(layer, 'rate'):
            layer_description = f"{str(layer)}"
        else:
            layer_description = f"Attivazione {str(layer_type)}"

        print(f"{Fore.CYAN}{i}: {layer_description}")
        print(f"{Fore.CYAN + '-' * 40}")
    print(f"\n{Fore.CYAN} {str(model_opt)}")

    print(Style.RESET_ALL)


def k_fold_indices(data, k):  # k = number of folds
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds


def k_fold_cross_validation(X, y, number_of_folds, n_output, layer_neurons, activation_function, regularizer, optimizer,
                            use_dropout, number_of_combinations, combination, epochs):
    # Liste per salvare i risultati
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_score_scores = []
    loss_scores = []  # Aggiunto per la regressione

    # Ottieni gli indici per i vari fold
    fold_indices = k_fold_indices(X, number_of_folds)
    print_check = False

    # Itera sui fold
    for i, (train_indices, test_indices) in enumerate(fold_indices):
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # Creazione del modello
        model = model_creation(X_train.shape[1], n_output, layer_neurons, activation_function[0], regularizer, optimizer, use_dropout)

        # Stampa il modello solo una volta
        if print_check is False:
            print_model(-1, model, combination, number_of_combinations)
            print_check = True

        if n_output == 1:
            task_type = "regression"
        else:
            task_type = "classification"
        print(f"\n========== Combination {i + 1} ==========\n")
        start_time = time.time()

        # Addestramento del modello
        model.train(X_train, y_train, epochs=epochs, batch_size=128, print_every=100, task_type=task_type)
        end_time = time.time()
        print("Tempo impiegato per il training di questo modello: ", end_time - start_time, "secondi")

        # Previsioni sui dati di test
        y_pred = model.predict(X_test)

        # Task di regressione
        if n_output == 1:
            loss = model.loss.calculate(y_pred, y_test)
            loss_scores.append(loss)

        # Task di classificazione
        else:
            accuracy, precision, recall, f1_score = compare_test_multiclass(y_pred, y_test)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_score_scores.append(f1_score)

    # Calcola la media dei risultati
    if n_output == 1:  # Regressione
        loss_mean = np.mean(loss_scores)
        print("\n========== K-Fold Cross-Validation Result (Regressione) ==========\n ")
        print("Mean Loss: ", loss_mean)
        return model, loss_mean

    else:  # Classificazione
        accuracy_mean = np.mean(accuracy_scores)
        precision_mean = np.mean(precision_scores)
        recall_mean = np.mean(recall_scores)
        f1_mean = np.mean(f1_score_scores)

        print("\n========== K-Fold Cross-Validation Result (Classificazione) ==========\n ")
        print("Mean Accuracy: ", accuracy_mean)
        print("Mean Precision: ", precision_mean)
        print("Mean Recall: ", recall_mean)
        print("Mean F1_Score: ", f1_mean)

        return model, accuracy_mean, precision_mean, recall_mean, f1_mean


def model_creation(X_train_shape, n_output, layer_neurons, activation_function, regularizer, optimizer, use_dropout):
    initialization = ''
    act_func_1 = None
    act_func_2 = None
    loss_function = None
    accuracy_metric = None
    if activation_function == 'relu':
        act_func_1 = Relu()
        act_func_2 = Relu()
        initialization = "He"
    else:
        act_func_1 = Tanh()
        act_func_2 = Tanh()
        initialization = "Glorot"
    if n_output >= 2:
        output_function = Softmax()
        loss_function = LossCategoricalCrossEntropy()
        accuracy_metric = AccuracyCategorical()
        modality = "max"
    else:
        output_function = ActivationLinear()
        loss_function = Rmse()
        accuracy_metric = None
        modality = "min"

    model = Model()
    if regularizer == "l1":

        model.add_layer(
            DenseLayer(X_train_shape, layer_neurons[0], initialization=initialization, l1_regularization_weights=0.001,
                       l1_regularization_bias=0.001))
        model.add_layer(act_func_1)
        check_dropout(model, use_dropout)
        model.add_layer(DenseLayer(layer_neurons[0], layer_neurons[1], initialization=initialization,
                                   l1_regularization_weights=0.001, l1_regularization_bias=0.001))
        model.add_layer(act_func_2)
        check_dropout(model, use_dropout)

        model.add_layer(
            DenseLayer(layer_neurons[1], n_output, initialization=initialization, l1_regularization_weights=0.001,
                       l1_regularization_bias=0.001))
        model.add_layer(output_function)

    elif regularizer == "l2":
        model.add_layer(
            DenseLayer(X_train_shape, layer_neurons[0], initialization=initialization, l2_regularization_weights=0.001,
                       l2_regularization_bias=0.001))
        model.add_layer(act_func_1)
        check_dropout(model, use_dropout)

        model.add_layer(DenseLayer(layer_neurons[0], layer_neurons[1], initialization=initialization,
                                   l2_regularization_weights=0.001, l2_regularization_bias=0.001))
        model.add_layer(act_func_2)
        check_dropout(model, use_dropout)
        model.add_layer(
            DenseLayer(layer_neurons[1], n_output, initialization=initialization, l2_regularization_weights=0.001,
                       l2_regularization_bias=0.001))
        model.add_layer(output_function)

    else:
        # No regularization (None)
        model.add_layer(DenseLayer(X_train_shape, layer_neurons[0], initialization=initialization))
        model.add_layer(act_func_1)
        check_dropout(model, use_dropout)
        model.add_layer(DenseLayer(layer_neurons[0], layer_neurons[1], initialization=initialization))
        model.add_layer(act_func_2)
        check_dropout(model, use_dropout)
        model.add_layer(DenseLayer(layer_neurons[1], n_output, initialization=initialization))
        model.add_layer(output_function)

    opt_alg = None

    if optimizer == "adam":
        opt_alg = Adam(decay=0.0001, learning_rate=0.01)
    elif optimizer == "rmsprop":
        opt_alg = Rmsprop(decay=0.0001, learning_rate=0.01)
    elif optimizer == "adagrad":
        opt_alg = Adagrad(learning_rate=0.01)
    elif optimizer == "sgd":
        opt_alg = Sgd(decay=0.0001, learning_rate=0.01)
    elif optimizer == "sgd_momentum":
        opt_alg = Sgd(decay=0.0001, momentum=0.9, learning_rate=0.01)

    model.set(
        loss=loss_function,
        optimizer=opt_alg,
        accuracy=accuracy_metric,
        early_stopping=EarlyStopping(patience=6, min_delta=0.001, mode=modality)
    )

    model.finalize()
    return model


layer_combination = [[512, 256], [256, 128], [1024, 512]]
regularizers = ["l2", "l1", None]
optimizers = ["adam", "rmsprop", "sgd_momentum"]
dropout = [True, False]
activation_functions = ["relu", "tanh"]

model_printed = False
thread_colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]


def parallel_train_fold(train_indices, test_indices, X, y, n_output, layer_neurons, activation_function, regularizer,
                        optimizer, use_dropout, combination, number_of_combinations, fold_number, epochs):
    global model_printed
    # Assegna un colore al thread
    thread_color = thread_colors[fold_number % len(thread_colors)]

    # Separa i dati in train e test per questo fold
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Crea il modello per questo fold
    model = model_creation(X_train.shape[1], n_output, layer_neurons, activation_function[0], regularizer, optimizer,
                           use_dropout)

    # Stampa la struttura del modello solo una volta
    if not model_printed:
        with threading.Lock():  # Blocco per sincronizzare i thread
            if not model_printed:
                print_model(-1, model, combination, number_of_combinations)
                model_printed = True

    # Stampa il fold corrente con il colore del thread
    print(f"{thread_color}== Combination {fold_number + 1} ==")

    if n_output == 1:
        task_type = "regression"
    else:
        task_type = "classification"

    # Esegui il training del modello
    start_time = time.time()
    model.train(X_train, y_train, epochs=epochs, batch_size=64, print_every=100, task_type=task_type)
    end_time = time.time()

    # Stampa il tempo impiegato per il training con il colore del thread
    print(
        f"{thread_color}Tempo impiegato per il training del modello del fold {fold_number + 1}: {end_time - start_time:.2f} secondi")

    # Previsioni sui dati di test
    y_pred = model.predict(X_test)

    if n_output >= 2:
        # Calcola le metriche di valutazione
        accuracy, precision, recall, f1_score = compare_test_multiclass(y_pred, y_test)
        return model, accuracy, precision, recall, f1_score
    else:
        loss = model.loss.calculate(y_pred, y_test)
        return model, loss


def k_fold_cross_validation_multithread(X, y, number_of_folds, n_output, layer_neurons, activation_function,
                                        regularizer, optimizer, use_dropout, number_of_combinations, combination,
                                        epochs):
    # Genera gli indici per i vari fold
    fold_indices = k_fold_indices(X, number_of_folds)

    # Inizializza le liste per memorizzare i risultati
    loss_score = []
    accuracy_score = []
    precision_score = []
    recall_score = []
    f1_score_score = []

    # Utilizza ProcessPoolExecutor per parallelizzare i vari fold
    with ProcessPoolExecutor() as executor:
        # Creiamo i task per ciascun fold
        futures = [
            executor.submit(parallel_train_fold, train_indices, test_indices, X, y, n_output, layer_neurons,
                            activation_function, regularizer, optimizer, use_dropout, combination,
                            number_of_combinations, fold_number, epochs)
            for fold_number, (train_indices, test_indices) in enumerate(fold_indices)
        ]

        # Recuperiamo i risultati
        for future in futures:
            result = future.result()

            if n_output == 1:
                model, loss = result
                loss_score.append(loss)

                # Task di classificazione
            else:
                model, accuracy, precision, recall, f1_score = result
                accuracy_score.append(accuracy)
                precision_score.append(precision)
                recall_score.append(recall)
                f1_score_score.append(f1_score)

                # Appendiamo i risultati per questo fold
                accuracy_score.append(accuracy)
                precision_score.append(precision)
                recall_score.append(recall)
                f1_score_score.append(f1_score)

    if n_output == 1:  # Regressione
        loss_mean = np.mean(loss_score)
        print("\n========== Risultati della K-Fold Cross-Validation per Regressione ==========\n")
        print(f"Mean Loss: {loss_mean}")
        return model, loss_mean

    else:  # Classificazione
        accuracy_mean = np.mean(accuracy_score)
        precision_mean = np.mean(precision_score)
        recall_mean = np.mean(recall_score)
        f1_mean = np.mean(f1_score_score)

        print("\n========== Risultati della K-Fold Cross-Validation per Classificazione ==========\n")
        print(f"Mean Accuracy: {accuracy_mean}")
        print(f"Mean Precision: {precision_mean}")
        print(f"Mean Recall: {recall_mean}")
        print(f"Mean F1_Score: {f1_mean}")

        return model, accuracy_mean, precision_mean, recall_mean, f1_mean


def validation(X_train, y_train, n_output, number_of_folders, epochs, multithread=False):
    all_combinations = product(layer_combination, activation_functions, regularizers, optimizers, dropout)
    i = 1
    start_time = time.time()
    for combination in all_combinations:
        numbers_of_neurons, activation_function, regularizer, optimizer, use_dropout = combination

        if multithread is False:
            if n_output == 1:
                model, loss_mean = k_fold_cross_validation(X_train, y_train, number_of_folders, n_output,
                                                           numbers_of_neurons, activation_function, regularizer,
                                                           optimizer, use_dropout, calculate_combinations_count(), i,
                                                           epochs)
            else:
                model, accuracy, precision, recall, f1_score = k_fold_cross_validation(X_train, y_train,
                                                                                       number_of_folders, n_output,
                                                                                       numbers_of_neurons,
                                                                                       activation_function, regularizer,
                                                                                       optimizer, use_dropout,
                                                                                       calculate_combinations_count(),
                                                                                       i, epochs)
        else:
            if n_output == 1:
                model, loss_mean = k_fold_cross_validation_multithread(X_train, y_train, number_of_folders, n_output,
                                                                       numbers_of_neurons, activation_function,
                                                                       regularizer, optimizer, use_dropout,
                                                                       calculate_combinations_count(), i, epochs)
            else:
                model, accuracy, precision, recall, f1_score = k_fold_cross_validation_multithread(X_train, y_train,
                                                                                                   number_of_folders,
                                                                                                   n_output,
                                                                                                   numbers_of_neurons,
                                                                                                   activation_function,
                                                                                                   regularizer,
                                                                                                   optimizer,
                                                                                                   use_dropout,
                                                                                                   calculate_combinations_count(),
                                                                                                   i, epochs)

        i += 1

        # Aggiorna il miglior modello in base alla loss (regressione)
        if n_output == 1 and loss_mean < best_model['loss']:
            best_model['loss'] = loss_mean
            best_model['model'] = model

        # Aggiorna il miglior modello in base a f1_score e altre metriche (classificazione)
        elif n_output > 1 and (f1_score > best_model['f1_metric'] or
                               (f1_score == best_model['f1_metric'] and precision > best_model['precision']) or
                               (f1_score == best_model['f1_metric'] and precision == best_model[
                                   'precision'] and recall > best_model['recall']) or
                               (f1_score == best_model['f1_metric'] and precision == best_model[
                                   'precision'] and recall == best_model['recall'] and accuracy > best_model[
                                    'accuracy'])):
            best_model['f1_metric'] = f1_score
            best_model['accuracy'] = accuracy
            best_model['precision'] = precision
            best_model['recall'] = recall
            best_model['model'] = model

        # Stampa i risultati migliori finora
        print("\n========== Best Model ==========\n")
        if n_output == 1:
            print(f"Loss: {best_model['loss']}")
        else:
            print(f"Accuracy: {best_model['accuracy']}")
            print(f"Precision: {best_model['precision']}")
            print(f"Recall: {best_model['recall']}")
            print(f"F1 Score: {best_model['f1_metric']}")

    end_time = time.time()
    print("Tempo impiegato per la C-V: ", end_time - start_time, "secondi")

    return best_model


def calculate_combinations_count():
    # Calcola il numero totale di combinazioni
    all_combinations = list(product(layer_combination, activation_functions, regularizers, optimizers, dropout))
    return len(all_combinations)


def check_dropout(model, dropout_bool):
    if dropout_bool is True:
        model.add_layer(Dropout(0.2))
