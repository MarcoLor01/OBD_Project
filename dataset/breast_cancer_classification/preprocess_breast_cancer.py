import pandas as pd
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
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # Convertire y_train, y_valid e y_test in array numpy e applicare reshape
    y_train = y_train.values.reshape(-1, 1)
    y_valid = y_valid.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)


    # Controllare le forme per verificare che corrispondano
    print(f"Forma di y_train_encoded: {y_train.shape}")
    print(f"Forma di y_valid_encoded: {y_valid.shape}")
    print(f"Forma di y_test_encoded: {y_test.shape}")

    return X_train, X_test, y_train, y_test, X_valid, y_valid


# Per regressione:

# Convertire y_train e y_val in array bidimensionali
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
