from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def california_housing():
    # Carica il dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    plt.figure(figsize=(10, 6))
    sns.histplot(y_train, bins=30, kde=True)
    plt.xlabel('Prezzo Mediano delle Case (y)')
    plt.title('Distribuzione dei Prezzi Mediani delle Case')
    plt.show()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, y_train, X_test, y_test
