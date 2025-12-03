
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from my_knn_project.simple_knn import SimpleKNN

import os

@pytest.fixture(scope="module")
def preprocessed_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dataset_path = os.path.join(project_root, 'src', 'my_knn_project', 'dataset.csv')
    
    dataset = pd.read_csv(dataset_path)

    dataset['TotalCharges'] = dataset['TotalCharges'].replace(" ", np.nan).astype(float)
    dataset['SeniorCitizen'] = dataset['SeniorCitizen'].astype(bool)
    dataset['Churn'] = dataset['Churn'].astype(bool)

    encoders = {}
    for col in dataset.columns:
        if dataset[col].isnull().any():
            fill_value = dataset[col].median() if dataset[col].dtype=='float' else dataset[col].mode()[0]
            dataset[col] = dataset[col].fillna(fill_value)
        if dataset[col].dtype=='object':
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

    all_columns = dataset.columns.values
    columns_without_churn = np.delete(all_columns, np.where(all_columns=='Churn'))
    scaler = StandardScaler()
    dataset[columns_without_churn] = scaler.fit_transform(dataset[columns_without_churn])

    X = dataset.drop('Churn', axis=1).values
    y = dataset['Churn'].astype(int).values

    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_predict_proba_shape(preprocessed_data):
    X_train, X_test, y_train, y_test = preprocessed_data
    knn = SimpleKNN(n_neighbors=5)
    knn.fit(X_train, y_train)

    probs = knn.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2), "predict_proba shape incorrect"
    assert np.allclose(probs.sum(axis=1), 1.0), "predict_proba rows must sum to 1"

def test_predict_labels_valid(preprocessed_data):
    X_train, X_test, y_train, y_test = preprocessed_data
    knn = SimpleKNN(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert np.all(np.isin(y_pred, [0, 1])), "Predicted labels must be 0 or 1"

def test_algorithm_consistency(preprocessed_data):
    X_train, X_test, y_train, y_test = preprocessed_data
    preds = {}
    for algo in ['brute', 'kd_tree', 'ball_tree']:
        knn = SimpleKNN(n_neighbors=5, algorithm=algo)
        knn.fit(X_train, y_train)
        preds[algo] = knn.predict(X_test)

    agreement_kd = np.mean(preds['brute'] == preds['kd_tree'])
    agreement_ball = np.mean(preds['brute'] == preds['ball_tree'])
    print(f"Agreement brute vs kd_tree: {agreement_kd:.4f}")
    print(f"Agreement brute vs ball_tree: {agreement_ball:.4f}")

    assert agreement_kd > 0.92, "brute vs kd_tree agreement too low"
    assert agreement_ball > 0.92, "brute vs ball_tree agreement too low"

def test_vs_sklearn_metrics(preprocessed_data):
    X_train, X_test, y_train, y_test = preprocessed_data

    my_knn = SimpleKNN(n_neighbors=5)
    my_knn.fit(X_train, y_train)
    y_pred_my = my_knn.predict(X_test)
    y_proba_my = my_knn.predict_proba(X_test)[:,1]

    sk_knn = KNeighborsClassifier(n_neighbors=5)
    sk_knn.fit(X_train, y_train)
    y_pred_sk = sk_knn.predict(X_test)
    y_proba_sk = sk_knn.predict_proba(X_test)[:,1]

    acc_my = accuracy_score(y_test, y_pred_my)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    f1_my = f1_score(y_test, y_pred_my)
    f1_sk = f1_score(y_test, y_pred_sk)
    roc_my = roc_auc_score(y_test, y_proba_my)
    roc_sk = roc_auc_score(y_test, y_proba_sk)

    print("\nMetrics comparison:")
    print(f"Accuracy - My KNN: {acc_my:.4f}, sklearn: {acc_sk:.4f}")
    print(f"F1 Score  - My KNN: {f1_my:.4f}, sklearn: {f1_sk:.4f}")
    print(f"ROC AUC   - My KNN: {roc_my:.4f}, sklearn: {roc_sk:.4f}")

    agreement = np.mean(y_pred_my == y_pred_sk)
    assert agreement > 0.7, f"Agreement with sklearn too low: {agreement:.2f}"

if __name__ == "__main__":
    pytest.main()
