# main_my_knn.py
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from simple_knn import SimpleKNN as knn

__author__ = "Damian"
__copyright__ = "Damian"
__license__ = "MIT"

def main():
    # Wczytanie danych
    dataset = pd.read_csv('./dataset.csv')
    print("Dataset loaded successfully.")

    # Typy kolumn
    dataset['TotalCharges'] = dataset['TotalCharges'].replace(" ", np.nan).astype(float)
    dataset['SeniorCitizen'] = dataset['SeniorCitizen'].astype(bool)
    dataset['Churn'] = dataset['Churn'].astype(bool)

    # Uzupełnienie braków i kodowanie
    encoders = {}
    for col in dataset.columns:
        if dataset[col].isnull().any():
            fill_value = dataset[col].median() if dataset[col].dtype == 'float' else dataset[col].mode()[0]
            dataset[col] = dataset[col].fillna(fill_value)
        if dataset[col].dtype == 'object':
            if dataset[col].nunique() > 2:
                ohe = OneHotEncoder()
                transformed = ohe.fit_transform(dataset[[col]])
                dataset = dataset.drop(col, axis=1)
                dataset = pd.concat([dataset, pd.DataFrame(transformed.toarray(), columns=ohe.get_feature_names_out([col]))], axis=1)
                encoders[col] = ohe
            else:
                le = LabelEncoder()
                dataset[col] = le.fit_transform(dataset[col])
                encoders[col] = le

    # Standaryzacja cech
    features = dataset.drop('Churn', axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = dataset['Churn'].astype(int).values

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    # Grid Search dla my_knn
    """
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'algorithm': ['brute', 'kd_tree', 'ball_tree'],
        'leaf_size': [20, 30]
    }

    grid = GridSearchCV(
        estimator=my_knn(),
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    my_knn = grid.best_estimator_
    """
    # Najlepsze parametry z grid search:{'algorithm': 'ball_tree', 'leaf_size': 20, 'metric': 'euclidean', 'n_neighbors': 15, 'weights': 'uniform'}
    my_knn=knn(n_neighbors=15, weights='uniform', metric='euclidean', algorithm='ball_tree', leaf_size=20)
    my_knn.fit(X_train, y_train)

    # Predykcja
    predicted_test_probs = my_knn.predict_proba(X_test)[:, 1]
    predicted_test_labels = my_knn.predict(X_test)

    # Metryki
    acc_my = sklearn.metrics.accuracy_score(y_test, predicted_test_labels)
    f1_my = sklearn.metrics.f1_score(y_test, predicted_test_labels)
    roc_my = sklearn.metrics.roc_auc_score(y_test, predicted_test_probs)

    print("\n=== my_knn ===")
    print(f'Accuracy score: {acc_my:.4f}')
    print(f'F1 score: {f1_my:.4f}')
    print(f'ROC AUC score: {roc_my:.4f}')

    # scikit-learn KNeighborsClassifier
    sk_knn = KNeighborsClassifier(n_neighbors=15, weights='uniform', metric='euclidean', algorithm='ball_tree', leaf_size=20)
    sk_knn.fit(X_train, y_train)

    y_pred_sk = sk_knn.predict(X_test)
    y_prob_sk = sk_knn.predict_proba(X_test)[:,1]

    acc_sk = sklearn.metrics.accuracy_score(y_test, y_pred_sk)
    f1_sk = sklearn.metrics.f1_score(y_test, y_pred_sk)
    roc_sk = sklearn.metrics.roc_auc_score(y_test, y_prob_sk)

    print("\n=== scikit-learn KNN ===")
    print(f'Accuracy score: {acc_sk:.4f}')
    print(f'F1 score: {f1_sk:.4f}')
    print(f'ROC AUC score: {roc_sk:.4f}')

    # Cross-validation
    cv_my = cross_val_score(my_knn, X, y, cv=5, scoring='accuracy')
    cv_sk = cross_val_score(sk_knn, X, y, cv=5, scoring='accuracy')

    print("\nCross-val accuracy:")
    print(f"Mymy_knn: {cv_my}, mean: {cv_my.mean():.4f}")
    print(f"SKlearn my_knn: {cv_sk}, mean: {cv_sk.mean():.4f}")

if __name__ == "__main__":
    main()