import numpy as np
from numpy._typing._array_like import NDArray
from typing import List, Tuple
from scipy.sparse import hstack
from data_preprocessing import BookPreprocessor
from data_preprocessing import SparseMatrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class MulticlassLogisticRegression:
    def __init__(
        
        self,
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
        regularization_strength: float = 0.01,
    ):
        """Initialize the model with hyperparameters."""
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_strength = regularization_strength

    def _softmax(self, z: SparseMatrix) -> np.ndarray:
        """ Compute the softmax of z.

        Args:
            z (SparseMatrix): 

        Returns:
            np.ndarray: 
        """
        z_dense = z.toarray()
        exp_z = np.exp(z_dense - z_dense.max(axis=1, keepdims=True))
        # Operands
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _cost(self, X: SparseMatrix, y: np.ndarray, theta: np.ndarray) -> float:
        """ Compute the cost function with L2 regularization.
        Args:
            X (SparseMatrix): x 
            y (np.ndarray): Y 
            theta (np.ndarray): weights and biases

        Returns:
            float: cost of the model
        """
        m = len(y)
        h = self._softmax(SparseMatrix(X @ theta))
        epsilon = 1e-5
        regularization_term = (self.regularization_strength / (2 * m)) * np.sum(
            np.square(theta[1:])
        )
        cost = (-1 / m) * np.sum(y * np.log(h + epsilon)) + regularization_term
        return cost

    def gradient_descent(
        self, X: SparseMatrix, y: np.ndarray, theta: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """ Perform gradient descent to learn theta.

        Args:
            X (SparseMatrix): X
            y (np.ndarray): Y
            theta (np.ndarray): weights and biases

        Returns:
            Tuple[np.ndarray, List[float]]: weights and cost_history for the step
        """
        m = len(y)
        cost_history = []

        for _ in range(self.num_iterations):
            h = self._softmax(SparseMatrix(X @ theta))
            gradient = (X.T @ (h - y) / m) + (self.regularization_strength / m) * np.r_[
                np.zeros((1, theta.shape[1])), theta[1:]
            ]
            theta -= self.learning_rate * gradient
            cost = self._cost(X, y, theta)
            cost_history.append(cost)

        return theta, cost_history

    def fit(
        self, X_train: SparseMatrix, y_train: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """ Train the model using the training data.

        Args:
            X_train (SparseMatrix): x
            y_train (np.ndarray): y

        Returns:
            Tuple[np.ndarray, List[float]]: theta and cost_history
        """
        m, n = X_train.shape
        k = len(np.unique(y_train))
        X_train = SparseMatrix(hstack([SparseMatrix(np.ones((m, 1))), X_train]))
        theta = np.zeros((n + 1, k))
        y_one_hot = np.eye(k)[y_train]

        theta, cost_history = self.gradient_descent(X_train, y_one_hot, theta)
        self.theta = theta
        return theta, cost_history

    def predict(self, X_test: SparseMatrix) -> np.ndarray:
        """ Predict the class labels for the test set.

        Args:
            X_test (SparseMatrix): X

        Returns:
            np.ndarray: predictions
        """
        m = X_test.shape[0]
        X_test = SparseMatrix(hstack([SparseMatrix(np.ones((m, 1))), X_test]))
        h = self._softmax(SparseMatrix(X_test @ self.theta))
        return np.argmax(h, axis=1)

    def evaluate(self, X_test: SparseMatrix, y_test: np.ndarray) -> dict:
        """ Evaluate the model on the test set.

        Args:
            X_test (SparseMatrix): X
            y_test (np.ndarray): Y

        Returns:
            dict: accuracy
        """
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test) * 100
        print(f"Accuracy: {accuracy:.2f}% 🎯")
        return {"accuracy": accuracy}

    def pretty_print_cost_history(self, cost_history):
        """Pretty print the cost history."""
        cost_df = pd.DataFrame(cost_history, columns=["Cost"])
        print(cost_df)
        plt.figure(figsize=(10, 6))
        plt.plot(cost_df.index, cost_df["Cost"], marker="o")
        plt.title("Cost History")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.show()

    def get_confusion_matrix(self, X_test, y) -> None:
        """
        Compute and print the confusion matrix of the model on the test set.
        Parameters:
            X: sparse matrix of shape (m, n) - Input features
            y: numpy array of shape (m,) - True labels (0 or 1)
            w: numpy array of shape (n,) - Weight vector
            b: float - Bias term
        Returns:
            None
        """
        # Get predictions
        predictions: NDArray = self.predict(X_test)
        # Compute confusion matrix
        cm = confusion_matrix(y_true=y, y_pred=predictions)
        # Plot confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Arthur Conan Doyle", "Fyodor Dostoyevsky", "Jane Austen"],
            yticklabels=["Arthur Conan Doyle", "Fyodor Dostoyevsky", "Jane Austen"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()


if __name__ == "__main__":
    # Example workflow
    data: BookPreprocessor = BookPreprocessor()
    split_data = data.get_data_splits()
    X_train, y_train = split_data["train"]
    X_val, y_val = split_data["val"]
    X_test, y_test = split_data["test"]

    # Convert to sparse matrices
    X_train = SparseMatrix(X_train)
    X_val = SparseMatrix(X_val)
    X_test = SparseMatrix(X_test)

    # Define range of lambda values to test
    # lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]
    # Perform cross-validation to select the best lambda
    # best_lambda: float | None = cross_validate_lambda(X_train, y_train, lambdas)
    # print(f"Best Lambda: {best_lambda} 🏆")
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
