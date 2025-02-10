from numpy._typing._array_like import NDArray
from numpy import float64
import numpy as np
from data_preprocessing import SMSSpamPreprocessor
from typing import Dict, Tuple


class LogisticRegression:
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
    ):
        """Initialize the Logistic Regression model with learning_rate and num_iterations."""
        self.learning_rate: float = learning_rate
        self.num_iterations: int = num_iterations
        self.regularization_strength: float = 0.05

    # Sigmoid function
    def _sigmoid(self, z: np.float64) -> np.float64:
        """Sigmoid function from example"""
        return 1 / (1 + np.exp(-z))

    def _cost(self, X, y, theta):
        """Log-loss function with L2 regularization"""
        m = len(y)
        h = self._sigmoid(np.dot(X, theta))
        epsilon = 1e-5
        regularization_term = (self.regularization_strength / (2 * m)) * np.sum(
            np.square(theta[1:])
        )
        return (-1 / m) * np.sum(
            y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)
        ) + regularization_term

    def gradient_descent(self, X, y, theta):
        """
        Perform batch gradient descent to learn theta with L2 regularization.

        Parameters:
            X: numpy array of shape (m, n) - Input features
            y: numpy array of shape (m,) - True labels (0 or 1)
            theta: numpy array of shape (n,) - Initial weight vector

        Returns:
            theta: numpy array of shape (n,) - Learned weight vector
            cost_history: list of float - Cost at each iteration
        """
        m = len(y)
        cost_history = []

        for _ in range(self.num_iterations):
            h = self._sigmoid(np.dot(X, theta))
            gradient = (np.dot(X.T, (h - y)) / m) + (
                self.regularization_strength / m
            ) * np.r_[0, theta[1:]]
            theta -= self.learning_rate * gradient
            cost = self._cost(X, y, theta)
            cost_history.append(cost)

        return theta, cost_history

    def stochastic_gradient_descent(self, X, y, theta):
        """
        Perform stochastic gradient descent to learn theta with L2 regularization.

        Parameters:
            X: numpy array of shape (m, n) - Input features
            y: numpy array of shape (m,) - True labels (0 or 1)
            theta: numpy array of shape (n,) - Initial weight vector

        Returns:
            theta: numpy array of shape (n,) - Learned weight vector
            cost_history: list of float - Cost at each iteration
        """
        m = len(y)
        cost_history = []

        for i in range(self.num_iterations):
            cost = 0
            for j in range(m):
                rand_index = np.random.randint(0, m)
                X_i = X[rand_index, :].reshape(1, X.shape[1])
                y_i = y[rand_index].reshape(1)
                h = self._sigmoid(np.dot(X_i, theta))
                gradient = (
                    np.dot(X_i.T, (h - y_i))
                    + (self.regularization_strength / m) * np.r_[0, theta[1:]]
                )
                theta -= self.learning_rate * gradient
                cost += self._cost(X_i, y_i, theta)
            cost_history.append(cost / m)

        return theta, cost_history

    def mini_batch_gradient_descent(self, X, y, theta, batch_size=32):
        """
        Perform mini-batch gradient descent to learn theta with L2 regularization.

        Parameters:
            X: numpy array of shape (m, n) - Input features
            y: numpy array of shape (m,) - True labels (0 or 1)
            theta: numpy array of shape (n,) - Initial weight vector
            batch_size: int - Size of each mini-batch

        Returns:
            theta: numpy array of shape (n,) - Learned weight vector
            cost_history: list of float - Cost at each iteration
        """
        m = len(y)
        cost_history = []

        for i in range(self.num_iterations):
            cost = 0
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for j in range(0, m, batch_size):
                X_i = X_shuffled[j : j + batch_size]
                y_i = y_shuffled[j : j + batch_size]
                h = self._sigmoid(np.dot(X_i, theta))
                gradient = (np.dot(X_i.T, (h - y_i)) / batch_size) + (
                    self.regularization_strength / m
                ) * np.r_[0, theta[1:]]
                theta -= self.learning_rate * gradient
                cost += self._cost(X_i, y_i, theta)
            cost_history.append(cost / (m // batch_size))

        return theta, cost_history

    def predict(self, X_pred: NDArray, w: NDArray, b: float) -> NDArray[float64]:
        """
        Predict the labels of the input features using the trained model.

        Parameters:
            X: numpy array of shape (m, n) - Input features
            w: numpy array of shape (n,) - Weight vector
            b: float - Bias term

        Returns:
            predictions: numpy array of shape (m,) - Predicted labels (0 or 1)
        """
        z = np.dot(X_pred, w) + b
        h = self._sigmoid(z)
        predictions = np.where(h >= 0.5, 1, 0)
        return predictions

    def fit(self, X_train, y, method="batch", batch_size=32):
        """
        Fit the logistic regression model to the training data.

        Parameters:
            X: numpy array of shape (m, n) - Input features
            y: numpy array of shape (m,) - True labels (0 or 1)
            method: str - Gradient descent method ('batch', 'stochastic', 'mini_batch')
            batch_size: int - Size of each mini-batch (used only for mini-batch gradient descent)

        Returns:
            w: numpy array of shape (n,) - Learned weight vector
            b: float - Learned bias term
        """
        X: NDArray[float64] = np.column_stack([np.ones(X_train.shape[0]), X_train])
        theta = np.zeros(X.shape[1])
        cost_history = []

        if method == "batch":
            theta, cost_history = self.gradient_descent(X, y, theta)
        elif method == "stochastic":
            theta, cost_history = self.stochastic_gradient_descent(X, y, theta)
        elif method == "mini_batch":
            theta, cost_history = self.mini_batch_gradient_descent(
                X, y, theta, batch_size
            )
        else:
            raise ValueError("Method must be 'batch', 'stochastic', or 'mini_batch'")

        w = theta[1:]
        b = theta[0]

        return w, b, cost_history

    def precision_score(self, y_true, y_pred):
        """
        Compute the precision score of the model.

        Parameters:
            y_true: numpy array of shape (m,) - True labels (0 or 1)
            y_pred: numpy array of shape (m,) - Predicted labels (0 or 1)

        Returns:
            precision: float - Precision score of the model
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        print(f"True Positives: {true_positives}")
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        print(f"False Positives: {false_positives}")
        precision = true_positives / (true_positives + false_positives)
        return precision

    def recall_score(self, y_true, y_pred):
        """
        Compute the recall score of the model.

        Parameters:
            y_true: numpy array of shape (m,) - True labels (0 or 1)
            y_pred: numpy array of shape (m,) - Predicted labels (0 or 1)

        Returns:
            recall: float - Recall score of the model
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        recall = true_positives / (true_positives + false_negatives)
        return recall

    def f1_score(self, precision, recall):
        """
        Compute the F1 score of the model.

        Parameters:
            y_true: numpy array of shape (m,) - True labels (0 or 1)
            y_pred: numpy array of shape (m,) - Predicted labels (0 or 1)

        Returns:
            f1: float - F1 score of the model
        """
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def evaluate(self, X_test, y, w, b) -> dict[str, float]:
        """
        Evaluate the performance of the model on the test set.

        Parameters:
            X: numpy array of shape (m, n) - Input features
            y: numpy array of shape (m,) - True labels (0 or 1)
            w: numpy array of shape (n,) - Weight vector
            b: float - Bias term

        Returns:
            accuracy: float - Accuracy of the model on the test set
        """
        # Get predictions
        predictions: NDArray = self.predict(X_pred=X_test, w=w, b=b)
        # Compute accuracy
        # Total Spam predictions (1)
        accuracy = np.mean(predictions == y) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        # Compute precision
        precision = self.precision_score(y_true=y, y_pred=predictions)
        print(f"Precision: {precision:.2f}")
        # Compute recall
        recall = self.recall_score(y_true=y, y_pred=predictions)
        print(f"Recall: {recall:.2f}")
        # Compute F1 score
        f1_score = self.f1_score(precision=precision, recall=recall)
        print(f"F1 Score: {f1_score:.2f}")
        # Return metrics
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }


if __name__ == "__main__":
    # Load the dataset
    data = SMSSpamPreprocessor("data/SMSSpamCollection")
    split_data: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = data.get_data_splits()
    X_train, y_train = split_data["train"]
    X_val, y_val = split_data["val"]
    X_test, y_test = split_data["test"]

    alpha = 1e-2
    max_iter = 1000

    lr = LogisticRegression(learning_rate=alpha, num_iterations=max_iter)
    w, b, cost_history = lr.fit(X_train, y_train)
    # Evaluate on validation set
    print("Validation set evaluation:")
    lr.evaluate(X_val, y_val, w, b)

    # Evaluate on test set
    print("Test set evaluation:")
    lr.evaluate(X_test, y_test, w, b)
