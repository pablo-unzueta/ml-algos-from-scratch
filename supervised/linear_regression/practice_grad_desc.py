import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


class LinearRegression:
    def __init__(
        self, lr: float = 0.1, tolerance: float = 1e-4, max_epochs: int = 1000
    ):
        if lr <= 0:
            raise ValueError(f"Learning Rate must be greater than 0\n{lr=}")
        self.lr = lr

        if tolerance <= 0:
            raise ValueError(f"Tolerance must be greater than 0\n{tolerance=}")
        self.tolerance = tolerance

        if max_epochs <= 0:
            raise ValueError(f"Max Epochs must be greater than 0\n{max_epochs=}")
        self.max_epochs = max_epochs

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_b = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.thetas = np.random.rand(X_b.shape[1], 1)
        loss_history, theta_history = [], []
        prev_loss = float("inf")

        for i in range(self.max_epochs):
            y_pred = np.dot(X_b, self.thetas)
            diff = y_pred - y
            loss = self.loss(diff)
            loss_history.append(loss)
            if abs(loss - prev_loss) < self.tolerance:
                break
            prev_loss = loss
            theta_history.append(self.thetas)
            self.thetas -= self.lr * self.gradients(X_b, diff)
        self.loss_history = loss_history
        self.theta_history = theta_history
        return self

    def loss(self, diff: np.ndarray):
        return np.mean(diff**2, axis=0)

    def gradients(self, X: np.ndarray, diff: np.ndarray):
        return np.dot(X.T, diff) / (2 * X.shape[0])

    def predict(self, X: np.ndarray):
        X_b = np.concatenate((np.ones_like(X), X), axis=1)
        return np.dot(X_b, self.thetas)


def plot(X: np.ndarray, y: np.ndarray):
    fig, ax = plt.subplots()
    ax.scatter(X, y, marker=".")
    plt.savefig("temp.png")


def main():
    X = np.random.rand(500, 1)
    y = 4 + 3 * X + np.random.randn(500, 1)
    X_test = np.random.rand(50, 1)
    y_test = 4 + 3 * X_test
    plot(X, y)
    lin_reg = LinearRegression()
    y_pred = lin_reg.fit(X, y).predict(X_test)
    print(y_pred - y_test)


if __name__ == "__main__":
    main()
