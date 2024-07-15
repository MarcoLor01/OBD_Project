from dataset.fashion_mnist_classification.preprocess_fashion_mnist import fashion_mnist_dataset
from dataset.fashion_mnist_classification.preprocess_fashion_mnist import shuffle_data
from dataset.fashion_mnist_classification.preprocess_fashion_mnist import total_step
from neural_network.randomizedSearch import randomizedSearchCV


X_train, y_train, X_test, y_test = fashion_mnist_dataset()
X_train, y_train = shuffle_data(X_train, y_train)
X_test, y_test = shuffle_data(X_test, y_test)

total_step_training = total_step(X_train)

randomized_search_parameters = {
    "metrics": ["AccuracyCategorical"],
    "optimizers": ["Adagrad", "Adam", "RmsProp", "Sgd"],
    "regularization": ["Dropout"],
    "number_of_layers": [2, 3, 4],
    "neurons_per_layer": [64, 128, 256, 512],
    "dropout_rate": [0.2, 0.3, 0.4, 0.5],
    "optimizer_params": {
        "Adagrad": [{"learning_rate": 0.01}, {"learning_rate": 0.001}],
        "Adam": [{"learning_rate": 0.001}, {"learning_rate": 0.0001}],
        "RmsProp": [{"learning_rate": 0.001}, {"learning_rate": 0.0001}],
        "Sgd": [{"learning_rate": 0.01}, {"learning_rate": 0.001}]
    }
}

randomizedSearchCV(X_train, y_test,
                   parameters=randomized_search_parameters,
                   combination=5,
                   number_of_classes=10
                   )

# # Instantiate the model
# model = Model()
# # Add layers
# model.add_layer(DenseLayer(X_train.shape[1], 64))
# model.add_layer(Relu())
# model.add_layer(DenseLayer(64, 64))
# model.add_layer(Relu())
# model.add_layer(DenseLayer(64, 10))
# model.add_layer(Softmax())
#
# model.set(
#     loss=LossCategoricalCrossEntropy(),
#     optimizer=Adam(decay=5e-5),
#     accuracy=AccuracyCategorical(),
#     early_stopping=EarlyStopping(patience=1, min_delta=0.0)
#     )
#
# model.finalize()
#
# model.train(X_train, y_train, val_data=(X_test, y_test), epochs=30, batch_size=128, print_every=100)
#
#
#
# confidences = model.predict(X_test[:20])
# predictions = model.output_activation.predictions(confidences)
# print(predictions)
# # Print first 5 labels
# print(y_test[:20])
