import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from neural_network.DenseLayer import DenseLayer
from neural_network.Model import Model
from neural_network.RandomizedSearch import print_model
from neural_network.activation_functions.LinearActivation import ActivationLinear
from neural_network.activation_functions.ReluActivationFunction import Relu
from neural_network.loss_functions.LossMSE import Mse
from neural_network.loss_functions.RMSE import Rmse
from neural_network.metrics_implementations.AccuracyRegression import AccuracyRegression
from neural_network.metrics_implementations.Mape import Mape
from neural_network.optimizers.Adagrad import Adagrad
from neural_network.optimizers.Adam import Adam
from neural_network.optimizers.RmsProp import Rmsprop
from neural_network.optimizers.Sgd import Sgd
from neural_network.regularization.Dropout import Dropout
from neural_network.regularization.EarlyStopping import EarlyStopping
from utils.Graphic import plot_residuals, graphic_regression_difference, plot_training_metrics
# Generazione del dataset semplice
np.random.seed(42)  # Per riproducibilità
X = np.random.rand(100, 1) * 10  # Input: numeri casuali tra 0 e 10
y = 2 * X + 1 + np.random.randn(100, 1)  # Funzione lineare con rumore

# Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Il mio modello
model = Model()
model.add_layer(DenseLayer(X_train.shape[1], 16, initialization="He"))
model.add_layer(Relu())
model.add_layer(DenseLayer(16, 32, initialization="He"))
model.add_layer(Relu())
model.add_layer(DenseLayer(32, 64, initialization="He"))
model.add_layer(Relu())
model.add_layer(DenseLayer(64, 1, initialization="He"))
model.add_layer(ActivationLinear())

model.set(loss=Mse(), optimizer=Adam(learning_rate=0.01), accuracy=AccuracyRegression())

model.finalize()
print_model(-1, model)
loss_history, accuracy_history, val_loss_history, val_accuracy_history = model.train(X_train, y_train, batch_size=32, val_data=(X_test, y_test), epochs=1, print_every=100)


confidences = model.predict(X_test)
mse = Rmse()
loss = mse.forward(confidences, y_test)

for i in range(len(y_test)):
    print("Etichetta: ", y_test[i], "Predizione: ", confidences[i])

plot_residuals(model, confidences, y_test)
graphic_regression_difference(y_test, confidences)
plot_training_metrics(loss_history, accuracy_history)
plot_training_metrics(val_loss_history, val_accuracy_history)


# Definizione del modello
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Uscita lineare per regressione
])

# Compilazione del modello
model.compile(optimizer='adam', loss='mean_squared_error')

# Addestramento del modello
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Valutazione sul test set
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Predizioni sui dati di test
predictions = model.predict(X_test)

# Creazione del grafico delle predizioni vs. valori reali
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Valori Reali')
plt.scatter(X_test, predictions, color='red', label='Predizioni del Modello')
plt.title("Predizioni vs Valori Reali")
plt.xlabel("Input X")
plt.ylabel("Target Y")
plt.legend()
plt.grid(True)
plt.show()
