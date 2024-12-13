import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)


class LinearRegression:
    def __init__(self, lr: float = 0.01, max_iter: int = 1000):
        if lr <= 0:
            raise ValueError(f"{lr=}\nLearning rate should be greater than 0")
        self.lr = lr

        if max_iter <= 0:
            raise ValueError(f"{max_iter}\nMax iterations should be greater than 0")
        self.max_iter = max_iter
        self.theta = np.random.randn(2, 1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same shape\nGot{X.shape=} and {y.shape=}"
            )

        X_b = np.concatenate((np.ones_like(X), X), axis=1)
        loss_history = []
        theta_history = []

        for i in range(self.max_iter):
            y_pred = np.dot(X_b, self.theta)
            diff = y_pred - y
            loss = self.loss_function(diff)
            gradients = X_b.T.dot(diff)
            loss_history.append(loss)
            theta_history.append(self.theta)
            self.theta -= gradients * self.lr

        self.loss_history = loss_history
        self.theta_history = theta_history

    def loss_function(self, diff: np.ndarray):
        return np.sum((diff) ** 2, axis=0) / diff.shape[0]

    def gradient(self, X: np.ndarray, diff: np.ndarray):
        return (2 / diff.shape[0] * np.dot(X.T, diff)).T

    def predict(self, X_test: np.ndarray):
        return X_test * self.theta[1] + self.theta[0]


def plot_data(
    X: np.ndarray,
    y: np.ndarray,
    path: Path,
    thetas: np.ndarray | None = None,
    fig_name: str = "plot.png",
):
    fig, ax = plt.subplots()
    if thetas is not None:
        plot_points = np.linspace(0, 1, 100)
        ax.plot(plot_points, plot_points * thetas[1] + thetas[0], color="red")
    ax.scatter(X, y)
    plt.savefig(path / fig_name)


def main():
    X = np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    lin_reg = LinearRegression(max_iter=100)
    lin_reg.fit(X, y)
    print(lin_reg.predict(X_test=np.array([0.5])))

    path = Path("/Users/pablo/software/ml-algos-from-scratch/supervised")
    plot_data(X, y, thetas=lin_reg.theta, path=path)


if __name__ == "__main__":
    main()
