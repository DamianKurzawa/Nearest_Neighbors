import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .simple_nn import SimpleNearestNeighbors


def visualize_k_vs_radius(X_train):
   
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_train)


    query_2d = np.array([[2.0, 0.0]])
    query = pca.inverse_transform(query_2d)

  
    knn = SimpleNearestNeighbors(n_neighbors=5)
    knn.fit(X_train)
    _, idx_k = knn.kneighbors(query)
    neighbors_k = X_2d[idx_k[0]]


    rnn = SimpleNearestNeighbors(radius=5.3)
    rnn.fit(X_train)
    _, idx_r = rnn.radius_neighbors(query)
    neighbors_r = X_2d[idx_r[0]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)


    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c="lightgray", s=20, label="Training data")
    axes[0].scatter(query_2d[0, 0], query_2d[0, 1],
                    c="red", s=120, marker="x", label="Query point")
    axes[0].scatter(neighbors_k[:, 0], neighbors_k[:, 1],
                    c="blue", s=80, label="k-NN")

    for p in neighbors_k:
        axes[0].plot([query_2d[0, 0], p[0]],
                     [query_2d[0, 1], p[1]],
                     c="blue", linestyle="--", alpha=0.4)

    axes[0].set_title("k-Nearest Neighbors")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].legend()

    axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c="lightgray", s=20, label="Training data")
    axes[1].scatter(query_2d[0, 0], query_2d[0, 1],
                    c="red", s=120, marker="x", label="Query point")
    axes[1].scatter(neighbors_r[:, 0], neighbors_r[:, 1],
                    c="green", s=80, label="Radius NN")

    for p in neighbors_r:
        axes[1].plot([query_2d[0, 0], p[0]],
                     [query_2d[0, 1], p[1]],
                     c="green", linestyle=":", alpha=0.4)

    axes[1].set_title("Radius Neighbors")
    axes[1].set_xlabel("PCA 1")
    axes[1].legend()

    plt.suptitle("k-Nearest Neighbors vs Radius Neighbors", fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_metrics_nn(X_train):
   
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_train)


    query_2d = np.array([[2.0, 0.0]])
    query = pca.inverse_transform(query_2d)

    metrics = [
        ("euclidean", None),
        ("manhattan", None),
        ("minkowski", 3),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    for ax, (metric, p) in zip(axes, metrics):
        nn = SimpleNearestNeighbors(
            n_neighbors=5,
            metric=metric,
            p=p if metric == "minkowski" else 2,
            algorithm="ball_tree"
        )
        nn.fit(X_train)

        _, idx = nn.kneighbors(query)
        neighbors_2d = X_2d[idx[0]]

 
        ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c="lightgray", s=20, label="Training data"
        )

        ax.scatter(
            query_2d[0, 0], query_2d[0, 1],
            c="red", s=120, marker="x", label="Query point"
        )

     
        ax.scatter(
            neighbors_2d[:, 0], neighbors_2d[:, 1],
            c="blue", s=80, label="Nearest neighbors"
        )


        for pnt in neighbors_2d:
            ax.plot(
                [query_2d[0, 0], pnt[0]],
                [query_2d[0, 1], pnt[1]],
                c="black", linestyle="--", alpha=0.4
            )

        title = metric.capitalize()
        if metric == "minkowski":
            title += " (p=3)"

        ax.set_title(title)
        ax.set_xlabel("PCA 1")

    axes[0].set_ylabel("PCA 2")
    axes[0].legend(loc="upper left")

    plt.suptitle("Nearest Neighbors – comparison of distance metrics (k=5)", fontsize=14)
    plt.tight_layout()
    plt.show()