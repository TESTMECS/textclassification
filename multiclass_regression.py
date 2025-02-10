import numpy as np
from numpy._typing._array_like import NDArray
from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from data_preprocessing import BookPreprocessor
from regression import cross_validate_lambda


class MulticlassLogisticRegression:
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
        regularization_strength: float = 0.01,
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_strength = regularization_strength

    def _softmax(self, z: csr_matrix) -> np.ndarray:
        """Compute the softmax of each row of the input z."""
        z_dense = z.toarray()
        exp_z = np.exp(z_dense - z_dense.max(axis=1, keepdims=True))
        # Operands 
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _cost(self, X: csr_matrix, y: np.ndarray, theta: np.ndarray) -> float:
        """Compute the cross-entropy loss with L2 regularization."""
        m = len(y)
        h = self._softmax(csr_matrix(X @ theta))
        epsilon = 1e-5
        regularization_term = (self.regularization_strength / (2 * m)) * np.sum(
            np.square(theta[1:])
        )
        cost = (-1 / m) * np.sum(y * np.log(h + epsilon)) + regularization_term
        return cost

    def gradient_descent(
        self, X: csr_matrix, y: np.ndarray, theta: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """Perform batch gradient descent to learn theta with L2 regularization."""
        m = len(y)
        cost_history = []

        for _ in range(self.num_iterations):
            h = self._softmax(csr_matrix(X @ theta))
            gradient = (X.T @ (h - y) / m) + (
                self.regularization_strength / m
            ) * np.r_[np.zeros((1, theta.shape[1])), theta[1:]]
            theta -= self.learning_rate * gradient
            cost = self._cost(X, y, theta)
            cost_history.append(cost)

        return theta, cost_history

    def fit(
        self, X_train: csr_matrix, y_train: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """Fit the multiclass logistic regression model to the training data."""
        m, n = X_train.shape
        k = len(np.unique(y_train))
        X_train = csr_matrix(hstack([csr_matrix(np.ones((m, 1))), X_train]))
        theta = np.zeros((n + 1, k))
        y_one_hot = np.eye(k)[y_train]

        theta, cost_history = self.gradient_descent(X_train, y_one_hot, theta)
        self.theta = theta
        return theta, cost_history

    def predict(self, X_test: csr_matrix) -> np.ndarray:
        """Predict the labels of the input features using the trained model."""
        m = X_test.shape[0]
        X_test = csr_matrix(hstack([csr_matrix(np.ones((m, 1))), X_test]))
        h = self._softmax(csr_matrix(X_test @ self.theta))
        return np.argmax(h, axis=1)

    def evaluate(self, X_test: csr_matrix, y_test: np.ndarray) -> dict:
        """Evaluate the performance of the model on the test set."""
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test) * 100
        print(f"Accuracy: {accuracy:.2f}% 🎯")
        return {"accuracy": accuracy}


if __name__ == "__main__":
    # Example workflow
    data = BookPreprocessor("data/books.txt")
    split_data: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = (
        data.get_data_splits()
    )
    X_train, y_train = split_data["train"]
    X_val, y_val = split_data["val"]
    X_test, y_test = split_data["test"]

    # Convert to sparse matrices
    X_train = csr_matrix(X_train)
    X_val = csr_matrix(X_val)
    X_test = csr_matrix(X_test)

    # Define range of lambda values to test
    lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]

    # Perform cross-validation to select the best lambda
    #best_lambda: float | None = cross_validate_lambda(X_train, y_train, lambdas)
    #print(f"Best Lambda: {best_lambda} 🏆")
    best_lambda = 0.01

    alpha = 0.01
    max_iter = 100

    # Train final model with the best lambda
    mclr = MulticlassLogisticRegression(
        learning_rate=alpha,
        num_iterations=max_iter,
        regularization_strength=best_lambda,
    )
    mclr.fit(X_train, y_train)

    # Evaluate on validation set
    print("Validation set evaluation: 📊")
    mclr.evaluate(X_val, y_val)

    # Evaluate on test set
    print("Test set evaluation: 📊")
    mclr.evaluate(X_test, y_test)
