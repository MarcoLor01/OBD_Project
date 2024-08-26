import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def flight_price_dataset():
    # Carica il dataset
    df = pd.read_csv('dataset/flight_price_prediction/flight_price.csv')

    # Trasformazione delle etichette
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    # Suddivisione con label
    X = df.drop(["price"], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Scaling dei dati
    mmscaler = MinMaxScaler(feature_range=(0, 1))

    # Fit scalatore solo su dati di training
    X_train = mmscaler.fit_transform(X_train)

    # Trasforma i dati di test con lo scalatore già fit sui dati di training
    X_test = mmscaler.transform(X_test)

    # Conversione in DataFrame per comodità
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    return X_train, y_train, X_test, y_test

