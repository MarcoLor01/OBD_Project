import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def flight_price_dataset():
    # Carica il dataset
    df = pd.read_csv('dataset/flight_price_prediction/flight_price.csv')
    df = df.dropna()

    # Trasformazione delle etichette
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    # Suddivisione con label
    X = df.drop(["price"], axis=1).values
    y = df['price'].values.reshape(-1, 1)  # Reshape di y per essere 2D

    # Divisione dei dati in training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling dei dati di input
    mmscaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = mmscaler_X.fit_transform(X_train)
    X_test = mmscaler_X.transform(X_test)

    # Scaling dei dati di target (y) #CONTROLLARE QUESTO SCALING
    mmscaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train = mmscaler_y.fit_transform(y_train)
    y_test = mmscaler_y.transform(y_test)

    return X_train, y_train, X_test, y_test, mmscaler_y

