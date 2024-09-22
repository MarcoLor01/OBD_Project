import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def breast_cancer_dataset():
    # Carica il dataset
    df = pd.read_csv('dataset/breast_cancer_classification/breast-cancer.csv')
    df = df.drop(columns=['id'], axis=1)
    pd.set_option('future.no_silent_downcasting', True)
    df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 0})

    # Rimuove le colonne con bassa correlazione
    df = df.drop(columns=['fractal_dimension_se', 'symmetry_se', 'texture_se', 'fractal_dimension_mean', 'smoothness_se'], axis=1)

    X = df.drop('diagnosis', axis=1)
    output = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, output, test_size=0.2, random_state=42, stratify=output)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convertire y_train, y_valid e y_test in array numpy e applicare reshape
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)


    return X_train, X_test, y_train, y_test


def convert_to_numeric_array(y):
    # Converti in array NumPy
    y_array = np.array(y)

    # Assicurati che l'array sia monodimensionale
    if y_array.ndim > 1:
        y_array = y_array.ravel()

    # Converti il tipo di dati a numerico, se possibile
    y_numeric = pd.to_numeric(y_array, errors='coerce')  # Usa pd.to_numeric per convertire a float
    y_numeric = np.nan_to_num(y_numeric, nan=0)  # Sostituisci NaN con 0
    y_numeric = y_numeric.astype(int)  # Converti a int

    return y_numeric

# Per regressione:

# Convertire y_train e y_val in array bidimensionali
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
