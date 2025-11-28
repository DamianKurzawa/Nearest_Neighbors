import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from my_knn_project.simple_knn import SimpleKNN

@pytest.fixture
def data():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                               n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.mark.parametrize("algorithm", ["brute", "kd_tree", "ball_tree"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev", "minkowski"])
@pytest.mark.parametrize("n_neighbors", [1, 3, 5])
def test_comparison_with_sklearn(data, algorithm, weights, metric, n_neighbors):
    X_train, X_test, y_train, y_test = data
    
    sk_kwargs = {
        "n_neighbors": n_neighbors,
        "algorithm": algorithm,
        "weights": weights,
        "metric": metric
    }
    
    my_kwargs = {
        "n_neighbors": n_neighbors,
        "algorithm": algorithm,
        "weights": weights,
        "metric": metric
    }
    
    sk_knn = KNeighborsClassifier(**sk_kwargs)
    sk_knn.fit(X_train, y_train)
    sk_pred = sk_knn.predict(X_test)
    
    my_knn = SimpleKNN(**my_kwargs)
    my_knn.fit(X_train, y_train)
    my_pred = my_knn.predict(X_test)
    
    agreement = np.mean(my_pred == sk_pred)
    

    assert agreement > 0.90, f"Low agreement for {algorithm}, {weights}, {metric}, k={n_neighbors}: {agreement}"

    if weights == 'uniform':
        sk_proba = sk_knn.predict_proba(X_test)
        my_proba = my_knn.predict_proba(X_test)

        
        if agreement == 1.0:
             np.testing.assert_allclose(my_proba, sk_proba, atol=1e-5, err_msg=f"Proba mismatch {algorithm} {metric} {n_neighbors}")
