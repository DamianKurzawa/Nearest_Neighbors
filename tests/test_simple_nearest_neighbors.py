import numpy as np
import pytest
from my_knn_project.simple_nn import SimpleNearestNeighbors


# =========================
# FIXTURES
# =========================

@pytest.fixture
def small_dataset():
    np.random.seed(0)
    return np.random.randn(20, 4)


@pytest.fixture
def query_points():
    return np.array([
        [0.1, -0.2, 0.3, -0.4],
        [1.0, 0.5, -0.3, 0.2]
    ])


# =========================
# BASIC FUNCTIONALITY
# =========================

def test_fit_stores_training_data(small_dataset):
    nn = SimpleNearestNeighbors()
    nn.fit(small_dataset)

    assert nn.X_train is not None
    assert nn.X_train.shape == small_dataset.shape


def test_kneighbors_shapes(small_dataset, query_points):
    nn = SimpleNearestNeighbors(n_neighbors=3)
    nn.fit(small_dataset)

    dist, idx = nn.kneighbors(query_points)

    assert dist.shape == (2, 3)
    assert idx.shape == (2, 3)


def test_kneighbors_sorted_by_distance(small_dataset):
    nn = SimpleNearestNeighbors(n_neighbors=5)
    nn.fit(small_dataset)

    dist, _ = nn.kneighbors([small_dataset[0]])

    assert np.all(dist[0][:-1] <= dist[0][1:])


# =========================
# DISTANCE METRICS
# =========================

@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "minkowski"])
def test_distance_metrics(metric, small_dataset):
    nn = SimpleNearestNeighbors(metric=metric, n_neighbors=3)
    nn.fit(small_dataset)

    dist, _ = nn.kneighbors([small_dataset[1]])

    assert np.all(dist >= 0)


# =========================
# EXCLUDE SELF
# =========================

def test_exclude_self_behavior(small_dataset):
    nn = SimpleNearestNeighbors(n_neighbors=3, exclude_self=True)
    nn.fit(small_dataset)

    dist, idx = nn.kneighbors(small_dataset)

    for i in range(len(small_dataset)):
        assert i not in idx[i]


# =========================
# RADIUS NEIGHBORS
# =========================

def test_radius_neighbors(small_dataset, query_points):
    nn = SimpleNearestNeighbors(radius=1.5)
    nn.fit(small_dataset)

    dist, idx = nn.radius_neighbors(query_points)

    assert len(dist) == len(query_points)
    assert len(idx) == len(query_points)


def test_radius_neighbors_with_small_radius(small_dataset):
    nn = SimpleNearestNeighbors(radius=0.01)
    nn.fit(small_dataset)

    dist, idx = nn.radius_neighbors([small_dataset[0]])

    assert len(dist[0]) <= 1


# =========================
# ALGORITHM CONSISTENCY
# =========================

def test_algorithm_consistency(small_dataset):
    results = []

    for algo in ["brute", "kd_tree", "ball_tree"]:
        nn = SimpleNearestNeighbors(n_neighbors=5, algorithm=algo)
        nn.fit(small_dataset)
        _, idx = nn.kneighbors([small_dataset[0]])
        results.append(idx)

    assert np.array_equal(results[0], results[1])
    assert np.array_equal(results[0], results[2])
