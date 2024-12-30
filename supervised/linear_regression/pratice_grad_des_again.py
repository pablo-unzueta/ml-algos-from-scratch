import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


class LinearRegression:
    def __init__(
        self,
        lr: float = 0.1,
        tol: float = 1e-8,
        max_epochs: int = 1000,
        thetas: np.ndarray = None,
        batch_size: int = None,
    ):
        if lr <= 0:
            raise ValueError(f"Learning rate must be greater than 0\n{lr=}")
        self.lr = lr

        if tol <= 0:
            raise ValueError(f"Tolerance must be greater than 0\n{tol=}")
        self.tol = tol

        if max_epochs <= 0:
            raise ValueError(f"Max epochs must be greater than 0\n{max_epochs=}")
        self.max_epochs = max_epochs

        self.thetas = thetas
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        # fix bias term
        X_b = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # X_b = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        if self.thetas is None:
            self.thetas = np.random.rand(X_b.shape[1], 1)
        elif self.thetas.shape != (X_b.shape[1], 1):
            raise ValueError(f"Thetas must be of shape {(X_b.shape[1], 1)}")

        prev_loss = float("inf")

        for epoch in range(self.max_epochs):
            if self.batch_size:
                indices = np.arange(X_b.shape[0])
                np.random.shuffle(indices)
                X_b = X_b[indices]
                y = y[indices]
                total_loss = 0

                for i in range(0, X_b.shape[0], self.batch_size):
                    X_batch = X_b[i : i + self.batch_size]
                    y_batch = y[i : i + self.batch_size]

                    y_pred = np.dot(X_batch, self.thetas)
                    diff = y_pred - y_batch
                    gradients = self.calc_gradients(X_batch, diff)
                    self.thetas -= self.lr * gradients
                    loss = self.loss_fn(diff)
                    total_loss += loss
                if abs(prev_loss - total_loss) <= self.tol:
                    break
                prev_loss = total_loss
            else:
                y_pred = np.dot(X_b, self.thetas)
                diff = y_pred - y
                loss = self.loss_fn(diff)
                if abs(prev_loss - loss) <= self.tol:
                    break
                gradients = self.calc_gradients(X_b, diff)
                self.thetas -= self.lr * gradients
                prev_loss = loss

            if epoch % 5 == 0:
                print(f"{loss=}")
        return self

    def loss_fn(self, diff: np.ndarray):
        # mse loss
        return np.mean((diff) ** 2, axis=0)

    def calc_gradients(self, X: np.ndarray, diff: np.ndarray):
        # 1/n 2 * X * (X \theta - y )
        return 2 * np.dot(X.T, diff) / diff.shape[0]

    def predict(self, X: np.ndarray):
        X_b = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.dot(X_b, self.thetas)


def plot(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.plot(X_test, y_test, color="red")
    plt.savefig("test_again.png")


def main():
    # create fake data
    X = np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    X_test = np.random.rand(50, 1)
    lin_reg = LinearRegression(lr=0.1, batch_size=2)
    y_test = lin_reg.fit(X, y).predict(X_test)
    print(f"{lin_reg.thetas=}")

    plot(X, y, X_test, y_test)


if __name__ == "__main__":
    main()
