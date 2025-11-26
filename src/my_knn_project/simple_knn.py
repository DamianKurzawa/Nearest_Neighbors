# simple_knn.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

__author__ = "Damian"
__copyright__ = "Damian"
__license__ = "MIT"

# ============================================================
#  KD-TREE (UPROSZCZONY, trzyma indeksy)
# ============================================================

class KDTree:
    def __init__(self, X, leaf_size=30):
        self.leaf_size = leaf_size
        self.data = np.asarray(X)
        self.n_features = self.data.shape[1] if len(self.data) > 0 else 0
        self.tree = self._build_tree(np.arange(len(self.data)), depth=0)

    def _build_tree(self, idx_array, depth):
        if idx_array.size == 0:
            return None
        if idx_array.size <= self.leaf_size:
            return {"leaf": True, "indices": idx_array}
        axis = depth % self.n_features
        # sort indices by axis and pick median index
        sorted_idx = idx_array[np.argsort(self.data[idx_array, axis])]
        median_pos = len(sorted_idx) // 2
        median_idx = sorted_idx[median_pos]
        median_value = self.data[median_idx, axis]
        left_idx = sorted_idx[:median_pos]
        right_idx = sorted_idx[median_pos + 1 :]
        return {
            "leaf": False,
            "axis": axis,
            "median_value": median_value,
            "index": median_idx,
            "left": self._build_tree(left_idx, depth + 1),
            "right": self._build_tree(right_idx, depth + 1),
        }

    def query(self, x, k, dist_fn):
        x = np.asarray(x).ravel()
        best = []  # list of tuples (distance, index)

        def consider_point(idx):
            d = dist_fn(self.data[idx:idx+1], x)[0]
            if len(best) < k:
                best.append((d, idx))
                best.sort(key=lambda t: t[0])
            else:
                if d < best[-1][0]:
                    best[-1] = (d, idx)
                    best.sort(key=lambda t: t[0])

        def search(node):
            if node is None:
                return
            if node["leaf"]:
                for idx in node["indices"]:
                    consider_point(int(idx))
                return

            axis = node["axis"]
            median_val = node["median_value"]

            # decide which side first
            if x[axis] <= median_val:
                first, second = node["left"], node["right"]
            else:
                first, second = node["right"], node["left"]

            # search nearer side
            search(first)

            # decide whether need to check the other side
            max_dist = best[-1][0] if best else np.inf
            # distance from point to splitting plane
            plane_dist = abs(x[axis] - median_val)
            if len(best) < k or plane_dist < max_dist:
                search(second)

            # also consider the median point itself (stored as 'index')
            consider_point(node["index"])

        search(self.tree)
        best.sort(key=lambda t: t[0])
        return best[:k]


# ============================================================
#  BALL-TREE (UPROSZCZONY, trzyma indeksy)
# ============================================================

class BallTree:
    def __init__(self, X, leaf_size=30):
        self.leaf_size = leaf_size
        self.data = np.asarray(X)
        self.tree = self._build_tree(np.arange(len(self.data)))

    def _build_tree(self, idx_array):
        if idx_array.size == 0:
            return None
        if idx_array.size <= self.leaf_size:
            return {"leaf": True, "indices": idx_array, "center": np.mean(self.data[idx_array], axis=0)}
        pts = self.data[idx_array]
        center = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        radius = np.max(dists)
        # split by farthest from center
        median = np.median(dists)
        left_idx = idx_array[dists <= median]
        right_idx = idx_array[dists > median]
        return {
            "leaf": False,
            "center": center,
            "radius": radius,
            "left": self._build_tree(left_idx),
            "right": self._build_tree(right_idx),
        }

    def query(self, x, k, dist_fn):
        x = np.asarray(x).ravel()
        best = []

        def consider_point(idx):
            d = dist_fn(self.data[idx:idx+1], x)[0]
            if len(best) < k:
                best.append((d, idx))
                best.sort(key=lambda t: t[0])
            else:
                if d < best[-1][0]:
                    best[-1] = (d, idx)
                    best.sort(key=lambda t: t[0])

        def search(node):
            if node is None:
                return
            if node["leaf"]:
                for idx in node["indices"]:
                    consider_point(int(idx))
                return
            # quick check using center/radius bound
            dist_center = np.linalg.norm(x - node["center"])
            # always search both if not enough neighbors
            if len(best) < k:
                search(node["left"])
                search(node["right"])
            else:
                # order children by which center is closer
                left = node["left"]
                right = node["right"]
                # If left or right is None, skip safely
                if left is not None and right is not None:
                    # try the child whose center is closer first
                    d_left = np.linalg.norm(x - left.get("center", node["center"]))
                    d_right = np.linalg.norm(x - right.get("center", node["center"]))
                    first, second = (left, right) if d_left <= d_right else (right, left)
                    search(first)
                    max_dist = best[-1][0]
                    # if possible overlap, search second
                    if (np.linalg.norm(x - second.get("center", node["center"])) - second.get("radius", 0)) <= max_dist:
                        search(second)
                else:
                    # fallback
                    search(left)
                    search(right)

        search(self.tree)
        best.sort(key=lambda t: t[0])
        return best[:k]


# ============================================================
#  SIMPLE KNN CLASSIFIER
# ============================================================

class SimpleKNN(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_neighbors=5,
                 weights="uniform",
                 algorithm="brute",
                 leaf_size=30,
                 metric="minkowski",
                 p=2):

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size

        self.metric = metric
        self.p = p

        self.X_train = None
        self.y_train = None
        self.classes_ = None

        self.tree = None   # KDTree or BallTree

    # -------------------------------------------------------
    # Distance function
    # -------------------------------------------------------
    def _dist(self, A, b):
        # A: array of shape (n_points, n_features), b: 1D array
        if self.metric == "euclidean":
            return np.sqrt(((A - b) ** 2).sum(axis=1))

        elif self.metric == "manhattan":
            return np.abs(A - b).sum(axis=1)

        elif self.metric == "chebyshev":
            return np.max(np.abs(A - b), axis=1)

        elif self.metric == "minkowski":
            return (np.abs(A - b) ** self.p).sum(axis=1) ** (1 / self.p)

        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    # -------------------------------------------------------
    # Fit
    # -------------------------------------------------------
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).astype(int)
        self.classes_ = np.unique(self.y_train)

        if self.algorithm == "kd_tree":
            self.tree = KDTree(self.X_train, leaf_size=self.leaf_size)
        elif self.algorithm == "ball_tree":
            self.tree = BallTree(self.X_train, leaf_size=self.leaf_size)
        else:
            self.tree = None  # brute-force

        return self

    # -------------------------------------------------------
    # Predict PROBA
    # -------------------------------------------------------
    def predict_proba(self, X):
        X = np.asarray(X)
        n_test = len(X)
        probs = np.zeros((n_test, len(self.classes_)))

        for i, x in enumerate(X):
            if self.tree is None:  # brute-force
                distances = self._dist(self.X_train, x)
                idx = np.argsort(distances)[:self.n_neighbors]
                neigh_labels = self.y_train[idx]
                neigh_dists = distances[idx]
            else:
                neighbors = self.tree.query(x, self.n_neighbors, lambda A, b: self._dist(A, b))
                # neighbors is list of (distance, index)
                if len(neighbors) == 0:
                    neigh_labels = np.array([], dtype=int)
                    neigh_dists = np.array([])
                else:
                    neigh_dists = np.array([d for d, idx in neighbors])
                    neigh_idx = np.array([idx for d, idx in neighbors], dtype=int)
                    neigh_labels = self.y_train[neigh_idx]

            if len(neigh_labels) == 0:
                # fallback: uniform random (shouldn't happen)
                probs[i] = np.ones(len(self.classes_)) / len(self.classes_)
                continue

            if self.weights == "uniform":
                w = np.ones(len(neigh_labels))
            elif self.weights == "distance":
                w = 1.0 / (neigh_dists + 1e-12)
            else:
                raise ValueError("weights must be 'uniform' or 'distance'")

            for idx_c, c in enumerate(self.classes_):
                probs[i, idx_c] = w[neigh_labels == c].sum()

            total = probs[i].sum()
            if total == 0:
                probs[i] = np.ones(len(self.classes_)) / len(self.classes_)
            else:
                probs[i] = probs[i] / total

        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # -------------------------------------------------------
    # sklearn API
    # -------------------------------------------------------
    def get_params(self, deep=True):
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "metric": self.metric,
            "p": self.p
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self