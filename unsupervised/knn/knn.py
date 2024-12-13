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

    def predict(self, query: np.ndarray):
        # Use Euclidean distances and sort
        print(self.data - query)
        # indices = np.argsort(np.linalg.norm(self.data - query))
        return self.labels[indices[: self.k]]


def plot_data(df: pd.DataFrame, path: Path, fig_name: str = "plot.png"):
    fig, ax = plt.subplots()
    colors = pd.factorize(df["Class"])[0]
    scatter_plot = ax.scatter(df["Weight(x2)"], df["Height(y2)"], c=colors)
    ax.set_xlabel("Weight")
    ax.set_ylabel("Height")
    plt.colorbar(scatter_plot)
    plt.savefig(path / fig_name)


if __name__ == "__main__":
    data_path = Path(
        "/Users/pablo/software/ml-algos-from-scratch/unsupervised/knn/data"
    )
    X = np.loadtxt(data_path / "data.txt", delimiter=",")
    y = np.loadtxt(data_path / "targets.txt")

    k_nearest_neighbors = kNN(k=3)
    k_nearest_neighbors.fit(X, y)

    x_test = np.array([1.0, 2.0])
    neighbors = k_nearest_neighbors.predict(x_test)
    print(f"{neighbors=}")
