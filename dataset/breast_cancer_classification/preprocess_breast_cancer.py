import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def breast_cancer_dataset():
    # Carica il dataset
    df = pd.read_csv('dataset/breast_cancer_classification/breast-cancer.csv')
    print(f"Il dataset contiene", df.shape[0], "righe e", df.shape[1], "colonne")
    df = df.drop(columns=['id'], axis=1)
    pd.set_option('future.no_silent_downcasting', True)
    df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 0})

    # Rimuove le colonne con bassa correlazione
    print("before", df.shape)
    df = df.drop(columns=['fractal_dimension_se', 'symmetry_se', 'texture_se', 'fractal_dimension_mean', 'smoothness_se'], axis=1)
    print("after", df.shape)

    X = df.drop('diagnosis', axis=1)
    output = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, output, test_size=0.2, random_state=42)
    plot_breast_cancer_distribution(y_train, y_test)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # Convertire y_train, y_valid e y_test in array numpy e applicare reshape
    y_train = y_train.values.reshape(-1, 1)
    y_valid = y_valid.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)


    return X_train, X_test, y_train, y_test, X_valid, y_valid


def plot_breast_cancer_distribution(y_train, y_test):
    # Unire i dati di train e test per visualizzare la distribuzione totale
    y_combined = np.concatenate([y_train, y_test])

    # Conta il numero di esempi per ogni classe (0 = benigno, 1 = maligno)
    unique, counts = np.unique(y_combined, return_counts=True)

    # Creare il grafico dell'istogramma
    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts, color=['#1f77b4', '#ff7f0e'])

    # Aggiungere etichette e titolo
    plt.title("Distribuzione di tumori maligni (1) e benigni (0) nel dataset")
    plt.xlabel("Diagnosi (0 = Benigno, 1 = Maligno)")
    plt.ylabel("Numero di istanze")

    # Aggiungere etichette alle barre
    for i, count in zip(unique, counts):
        plt.text(i, count + 5, str(count), ha='center', fontsize=12)

    # Mostrare il grafico
    plt.xticks([0, 1])  # Etichette degli assi x (0 e 1)
    plt.show()

# Per regressione:

# Convertire y_train e y_val in array bidimensionali
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
