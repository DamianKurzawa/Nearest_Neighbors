# main_nn.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from simple_nn import SimpleNearestNeighbors

__author__ = "Damian"
__license__ = "MIT"


def preprocess_dataset(path="./dataset.csv"):
    """Wczytanie, czyszczenie i kodowanie danych."""
    dataset = pd.read_csv(path)
    print("Dataset loaded successfully.")

    dataset['TotalCharges'] = dataset['TotalCharges'].replace(" ", np.nan).astype(float)
    dataset['SeniorCitizen'] = dataset['SeniorCitizen'].astype(bool)
    dataset['Churn'] = dataset['Churn'].astype(bool)

    # Uzupełnianie braków i kodowanie
    for col in dataset.columns:
        if dataset[col].isnull().any():
            fill_value = dataset[col].median() if dataset[col].dtype == float else dataset[col].mode()[0]
            dataset[col] = dataset[col].fillna(fill_value)

        if dataset[col].dtype == object:
            if dataset[col].nunique() > 2:
                ohe = OneHotEncoder()
                transformed = ohe.fit_transform(dataset[[col]])
                dataset = dataset.drop(col, axis=1)
                dataset = pd.concat(
                    [dataset, pd.DataFrame(transformed.toarray(),
                                           columns=ohe.get_feature_names_out([col]))],
                    axis=1
                )
            else:
                le = LabelEncoder()
                dataset[col] = le.fit_transform(dataset[col])

    # Standaryzacja
    X = dataset.drop("Churn", axis=1).values
    y = dataset["Churn"].astype(int).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def compare_nearest_neighbors(X_train, X_test):
    """Porównanie rezultatów odległości i indeksów NN."""

    print("\n==============================")
    print("  SimpleNearestNeighbors vs sklearn.NearestNeighbors")
    print("==============================\n")

    n_neighbors = 10

    # --- Twoja implementacja ---
    my_nn = SimpleNearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="ball_tree",
        metric="euclidean",
        leaf_size=20,
        exclude_self=False
    )
    my_nn.fit(X_train)

    my_d, my_idx = my_nn.kneighbors([X_test[0]])

    print("=== SimpleNearestNeighbors ===")
    print("Distances (first sample):", my_d[0])
    print("Indices   (first sample):", my_idx[0])

    # --- sklearn ---
    sk_nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="ball_tree",
        metric="euclidean",
        leaf_size=20
    )
    sk_nn.fit(X_train)

    sk_d, sk_idx = sk_nn.kneighbors([X_test[0]])

    print("\n=== sklearn.NearestNeighbors ===")
    print("Distances (first sample):", sk_d[0])
    print("Indices   (first sample):", sk_idx[0])



def main():
    X, y = preprocess_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    compare_nearest_neighbors(X_train, X_test)


if __name__ == "__main__":
    main()
