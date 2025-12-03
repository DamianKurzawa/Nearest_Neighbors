# simple_nn.py
import numpy as np
from sklearn.base import BaseEstimator

class KDTree:
    def __init__(self, X, leaf_size=30):
        self.leaf_size = leaf_size
        self.data = np.asarray(X)
        self.n_features = self.data.shape[1]
        self.tree = self._build_tree(np.arange(len(self.data)), depth=0)

    def _build_tree(self, idx_array, depth):
        if len(idx_array) == 0:
            return None
        if len(idx_array) <= self.leaf_size:
            return {"leaf": True, "indices": idx_array}

        axis = depth % self.n_features
        sorted_idx = idx_array[np.argsort(self.data[idx_array, axis])]
        median_pos = len(sorted_idx) // 2

        left = sorted_idx[:median_pos]
        right = sorted_idx[median_pos+1:]
        median_idx = sorted_idx[median_pos]

        return {
            "leaf": False,
            "axis": axis,
            "median": self.data[median_idx, axis],
            "left": self._build_tree(left, depth + 1),
            "right": self._build_tree(right, depth + 1),
            "median_idx": median_idx
        }

    def query(self, x, k, dist_fn):
        x = np.asarray(x).ravel()
        best = []

        def add_candidate(idx):
            d = dist_fn(self.data[idx:idx+1], x)[0]
            if len(best) < k:
                best.append((d, idx))
                best.sort(key=lambda t: t[0])
            else:
                if d < best[-1][0]:
                    best[-1] = (d, idx)
                    best.sort(key=lambda t: t[0])

        def search(node, depth=0):
            if node is None:
                return

            if node["leaf"]:
                for idx in node["indices"]:
                    add_candidate(int(idx))
                return

            axis = node["axis"]
            median_val = node["median"]

            go_left = x[axis] <= median_val

            first = node["left"] if go_left else node["right"]
            second = node["right"] if go_left else node["left"]

            search(first, depth+1)

            if len(best) < k or abs(x[axis] - median_val) < best[-1][0]:
                search(second, depth+1)

            add_candidate(node["median_idx"])

        search(self.tree)
        best.sort(key=lambda t: t[0])
        return best[:k]


class BallTree:
    def __init__(self, X, leaf_size=30):
        self.leaf_size = leaf_size
        self.data = np.asarray(X)
        self.tree = self._build_tree(np.arange(len(self.data)))

    def _build_tree(self, idx_array):
        if len(idx_array) == 0:
            return None
        if len(idx_array) <= self.leaf_size:
            return {
                "leaf": True,
                "indices": idx_array,
                "center": np.mean(self.data[idx_array], axis=0),
                "radius": 0
            }

        pts = self.data[idx_array]
        center = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        radius = np.max(dists)

        median = np.median(dists)
        left = idx_array[dists <= median]
        right = idx_array[dists > median]

        return {
            "leaf": False,
            "center": center,
            "radius": radius,
            "left": self._build_tree(left),
            "right": self._build_tree(right)
        }

    def query(self, x, k, dist_fn):
        x = np.asarray(x).ravel()
        best = []

        def add(idx):
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
                    add(int(idx))
                return

            dist_center = np.linalg.norm(x - node["center"])

            first = node["left"] if dist_center < node["radius"] else node["right"]
            second = node["right"] if first is node["left"] else node["left"]

            search(first)

            if len(best) < k or (dist_center - node["radius"]) < best[-1][0]:
                search(second)

        search(self.tree)
        best.sort(key=lambda t: t[0])
        return best[:k]


class SimpleNearestNeighbors(BaseEstimator):

    def __init__(self,
                 n_neighbors=5,
                 radius=1.0,
                 algorithm="brute",
                 leaf_size=30,
                 metric="minkowski",
                 p=2,
                 exclude_self=False):

        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.exclude_self = exclude_self

        self.X_train = None
        self.tree = None


    def _dist(self, A, b):
        if self.metric == "euclidean":
            return np.sqrt(((A - b) ** 2).sum(axis=1))
        elif self.metric == "manhattan":
            return np.abs(A - b).sum(axis=1)
        elif self.metric == "chebyshev":
            return np.max(np.abs(A - b), axis=1)
        elif self.metric == "minkowski":
            return (np.abs(A - b) ** self.p).sum(axis=1) ** (1 / self.p)
        else:
            raise ValueError("Unsupported metric")


    def fit(self, X):
        self.X_train = np.asarray(X)

        if self.algorithm == "kd_tree":
            self.tree = KDTree(self.X_train, leaf_size=self.leaf_size)
        elif self.algorithm == "ball_tree":
            self.tree = BallTree(self.X_train, leaf_size=self.leaf_size)
        else:
            self.tree = None

        return self


    def kneighbors(self, X_query, n_neighbors=None, return_distance=True):
        X_query = np.asarray(X_query)
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        distances_all = []
        indices_all = []

        for x in X_query:

            if self.tree is None:
                d = self._dist(self.X_train, x)
                order = np.argsort(d)

                self_index = np.argmin(self._dist(self.X_train, x))

                if self.exclude_self:
                    order = order[order != self_index]
                    
                order = order[:n_neighbors]

                distances_all.append(d[order])
                indices_all.append(order)


            else:
                neigh = self.tree.query(x, n_neighbors + 1, lambda A, b: self._dist(A, b))
                d = np.array([d for d, ii in neigh])
                idx = np.array([ii for d, ii in neigh], dtype=int)

                if self.exclude_self:
                    mask = idx != np.argmin(self._dist(self.X_train, x))
                    d = d[mask]
                    idx = idx[mask]

                d = d[:n_neighbors]
                idx = idx[:n_neighbors]

                distances_all.append(d)
                indices_all.append(idx)

        distances_all = np.vstack(distances_all)
        indices_all = np.vstack(indices_all)

        return (distances_all, indices_all) if return_distance else indices_all
    
 
    def radius_neighbors(self, X_query, radius=None, return_distance=True):
        X_query = np.asarray(X_query)
        if radius is None:
            radius = self.radius

        results_idx = []
        results_dist = []

        for x in X_query:
            d = self._dist(self.X_train, x)
            mask = d <= radius

            idx = np.where(mask)[0]
            dist = d[mask]

            if self.exclude_self:
                mask2 = idx != np.argmin(self._dist(self.X_train, x))
                idx = idx[mask2]
                dist = dist[mask2]

            results_idx.append(idx)
            results_dist.append(dist)

        if return_distance:
            return results_dist, results_idx
        return results_idx


