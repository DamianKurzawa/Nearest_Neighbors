# tests/test_simple_nearest_neighbors.py
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from my_knn_project.simple_nn import SimpleNearestNeighbors


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def small_dataset():
    np.random.seed(0)
    X = np.random.randn(50, 4)
    return X


@pytest.fixture
def query_points():
    return np.array([[0.1, -0.2, 0.3, -0.4],
                     [1.0,  0.5, -0.3, 0.2]])


# ---------------------------------------------------------
# Test FIT
# ---------------------------------------------------------

def test_fit_stores_training_data(small_dataset):
    model = SimpleNearestNeighbors()
    model.fit(small_dataset)

    assert model.X_train is not None
    assert model.X_train.shape == small_dataset.shape


# ---------------------------------------------------------
# Test get_params / set_params
# ---------------------------------------------------------

def test_params_roundtrip():
    model = SimpleNearestNeighbors(n_neighbors=7, algorithm="kd_tree", metric="manhattan")

    params = model.get_params()
    assert params["n_neighbors"] == 7
    assert params["algorithm"] == "kd_tree"
    assert params["metric"] == "manhattan"

    model.set_params(n_neighbors=3, metric="euclidean")
    assert model.n_neighbors == 3
    assert model.metric == "euclidean"


# ---------------------------------------------------------
# Test kneighbors shape and types
# ---------------------------------------------------------

def test_kneighbors_shapes(small_dataset, query_points):
    model = SimpleNearestNeighbors(n_neighbors=5)
    model.fit(small_dataset)

    distances, indices = model.kneighbors(query_points)

    assert distances.shape == (2, 5)
    assert indices.shape == (2, 5)


# ---------------------------------------------------------
# Test brute-force vs sklearn
# ---------------------------------------------------------

def test_bruteforce_matches_sklearn(small_dataset, query_points):
    my_nn = SimpleNearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    sk_nn = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")

    my_nn.fit(small_dataset)
    sk_nn.fit(small_dataset)

    d1, idx1 = my_nn.kneighbors(query_points)
    d2, idx2 = sk_nn.kneighbors(query_points)

    # distances may differ in order for ties, but values must match
    np.testing.assert_allclose(d1, d2, atol=1e-6)
    assert np.array_equal(idx1, idx2)


# ---------------------------------------------------------
# Test brute-force vs KDTree consistency
# ---------------------------------------------------------

def test_kd_tree_matches_bruteforce(small_dataset, query_points):
    brute_nn = SimpleNearestNeighbors(n_neighbors=5, algorithm="brute")
    kd_nn = SimpleNearestNeighbors(n_neighbors=5, algorithm="kd_tree")

    brute_nn.fit(small_dataset)
    kd_nn.fit(small_dataset)

    d1, idx1 = brute_nn.kneighbors(query_points)
    d2, idx2 = kd_nn.kneighbors(query_points)

    np.testing.assert_allclose(d1, d2, atol=1e-6)
    assert np.array_equal(idx1, idx2)


# ---------------------------------------------------------
# Test brute-force vs BallTree consistency
# ---------------------------------------------------------

def test_ball_tree_matches_bruteforce(small_dataset, query_points):
    brute_nn = SimpleNearestNeighbors(n_neighbors=5, algorithm="brute")
    ball_nn = SimpleNearestNeighbors(n_neighbors=5, algorithm="ball_tree")

    brute_nn.fit(small_dataset)
    ball_nn.fit(small_dataset)

    d1, idx1 = brute_nn.kneighbors(query_points)
    d2, idx2 = ball_nn.kneighbors(query_points)

    np.testing.assert_allclose(d1, d2, atol=1e-6)
    assert np.array_equal(idx1, idx2)


# ---------------------------------------------------------
# Test radius_neighbors
# ---------------------------------------------------------

def test_radius_neighbors(small_dataset, query_points):
    model = SimpleNearestNeighbors(radius=1.5)
    model.fit(small_dataset)

    dist, idx = model.radius_neighbors(query_points, radius=1.5)

    assert len(dist) == len(query_points)
    assert len(idx) == len(query_points)
    assert all(isinstance(arr, np.ndarray) for arr in idx)


# ---------------------------------------------------------
# Test exclude_self
# ---------------------------------------------------------

def test_exclude_self_behavior(small_dataset):
    X = small_dataset[:10]
    model = SimpleNearestNeighbors(n_neighbors=3, exclude_self=True)
    model.fit(X)

    d, idx = model.kneighbors(X)

    for i in range(len(X)):
        assert i not in idx[i]
