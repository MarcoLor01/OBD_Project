import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Scarica il dataset Boston Housing da URL
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

X = data.drop('medv', axis=1)  # Feature
y = data['medv'].values.reshape(-1, 1)  # Target
X = X.values
# Divisione dei dati in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
# Normalizzazione dei dati
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

model = Model()
model.add_layer(DenseLayer(X_train.shape[1], 16, initialization="He", l2_regularization_weights=0.01, l2_regularization_bias=0.01))
model.add_layer(Dropout(0.2))
model.add_layer(Relu())
model.add_layer(DenseLayer(16, 32, initialization="He", l2_regularization_weights=0.01, l2_regularization_bias=0.01))
model.add_layer(Dropout(0.2))
model.add_layer(Relu())
model.add_layer(DenseLayer(32, 64, initialization="He", l2_regularization_weights=0.01, l2_regularization_bias=0.01))
model.add_layer(Relu())
model.add_layer(DenseLayer(64, 1, initialization="He", l2_regularization_weights=0.01, l2_regularization_bias=0.01))
model.add_layer(ActivationLinear())

model.set(
    loss=Mse(),
    optimizer=Adam(learning_rate=0.01, decay=5e-5),
    accuracy=Mape(),
)

model.finalize()
print_model(-1, model)
loss_history, accuracy_history, val_loss_history, val_accuracy_history = model.train(X_train, y_train, batch_size=32, val_data=(X_valid, y_valid), epochs=300, print_every=100)


confidences = model.predict(X_test)
mse = Rmse()
loss = mse.forward(confidences, y_test)

for i in range(len(y_test)):
    print("Etichetta: ", y_test[i], "Predizione: ", confidences[i])

plot_residuals(model, confidences, y_test)
graphic_regression_difference(y_test, confidences)
plot_training_metrics(loss_history, accuracy_history)
plot_training_metrics(val_loss_history, val_accuracy_history)
