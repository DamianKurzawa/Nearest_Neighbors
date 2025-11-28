import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from my_knn_project.simple_knn import SimpleKNN

@pytest.fixture
def toy_data():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3,
                               n_redundant=0, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_fit_predict_shapes(toy_data):
    X_train, X_test, y_train, y_test = toy_data
    knn = SimpleKNN(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predict output shape mismatch"

    probs = knn.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2), "Predict_proba shape mismatch"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities do not sum to 1"

def test_brute_vs_tree_algorithms(toy_data):
    X_train, X_test, y_train, y_test = toy_data
    results = []
    for algo in ["brute", "kd_tree", "ball_tree"]:
        knn = SimpleKNN(n_neighbors=3, algorithm=algo)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        results.append(y_pred)
    assert np.array_equal(results[0], results[1]), "Brute vs KDTree mismatch"
    assert np.array_equal(results[0], results[2]), "Brute vs BallTree mismatch"

def test_weights(toy_data):
    X_train, X_test, y_train, y_test = toy_data
    for weight in ["uniform", "distance"]:
        knn = SimpleKNN(n_neighbors=3, weights=weight)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        assert np.all(np.isin(y_pred, [0, 1])), "Predictions contain invalid class"

def test_metric_distance_consistency(toy_data):
    X_train, X_test, y_train, y_test = toy_data
    for metric in ["euclidean", "manhattan", "chebyshev", "minkowski"]:
        knn = SimpleKNN(n_neighbors=3, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        assert np.all(np.isin(y_pred, [0, 1])), f"Predictions invalid for metric {metric}"

def test_vs_sklearn(toy_data):
    X_train, X_test, y_train, y_test = toy_data
    my_knn = SimpleKNN(n_neighbors=5)
    my_knn.fit(X_train, y_train)
    my_pred = my_knn.predict(X_test)

    sk_knn = KNeighborsClassifier(n_neighbors=5)
    sk_knn.fit(X_train, y_train)
    sk_pred = sk_knn.predict(X_test)

    common = np.mean(my_pred == sk_pred)
    assert common > 0.7, f"Too low agreement with sklearn: {common:.2f}"

if __name__ == "__main__":
    pytest.main()
