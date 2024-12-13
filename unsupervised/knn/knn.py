import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class kNN:
    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.data = X
        self.labels = y

    def predict(self, query: np.ndarray, method: str = "vectorized"):
        method_type = ["vectorized", "loop"]
        if method not in method_type:
            raise ValueError(f"{method=}. Must be vectorized or loop!")
        elif method == method_type[0]:
            # Use Euclidean distances and get indices
            indices = np.argsort(np.linalg.norm(self.data - query, axis=1))[: self.k]
        else:
            # Use for loop
            dist = []
            for point in self.data:
                dist.append(np.sum((point - query) ** 2))
            indices = sorted(range(len(dist)), key=lambda x: dist[x])
        return self.labels[indices][: self.k]


def plot_data(
    X: np.ndarray,
    labels: np.ndarray,
    path: Path,
    test_point: np.ndarray | None = None,
    fig_name: str = "plot.png",
):
    fig, ax = plt.subplots()
    if test_point is not None:
        ax.scatter(test_point[0], test_point[1], color="red", marker="*", s=100)
    scatter_plot = ax.scatter(X[:, 0], X[:, 1], c=labels)
    plt.colorbar(scatter_plot)
    plt.savefig(path / fig_name)


if __name__ == "__main__":
    data_path = Path(
        "/Users/pablo/software/ml-algos-from-scratch/unsupervised/knn/data"
    )
    X = np.loadtxt(data_path / "data.txt", delimiter=",")
    y = np.loadtxt(data_path / "targets.txt")

    k_nearest_neighbors = kNN(k=10)
    k_nearest_neighbors.fit(X, y)

    x_test = np.array([1.0, -1.0])
    plot_data(X=X, labels=y, path=data_path, test_point=x_test)
    neighbors = k_nearest_neighbors.predict(x_test, method="vectorized")
    print(f"{neighbors=}")
