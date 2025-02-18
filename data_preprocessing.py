import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from scipy.sparse import csr_matrix, issparse

import seaborn as sns
import matplotlib.pyplot as plt

# Typing
from numpy._typing._array_like import NDArray
from typing import Any, Tuple, Dict, TypeAlias, Union, List
from numpy._typing import NDArray

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Set random seed for reproducibility
np.random.seed(seed=420)

# Define type alias for sparse matrix
SparseMatrix: TypeAlias = csr_matrix


class TextPreprocessor:
    def __init__(
        self,
        data: List[str],
        labels: List[str],
        test_size: float = 0.2,
        val_size: float = 0.1,
        scaler=StandardScaler(with_mean=False),
        report: bool = False,
    ):
        """Initialize the TextPreprocessor with data and labels."""
        # Data and Labels
        self.data = data
        self.labels = labels
        # Train and test sets
        self.test_size = test_size
        self.val_size = val_size
        # Preprocessing
        self._vectorizer = TfidfVectorizer()
        self._stemmer = PorterStemmer()
        self._stop_words = stopwords.words("english")
        self._scaler = scaler
        # Report
        self.report = report

    def _report_data_distribution(self, _data, labels: pd.Series):
        """If Report, plot the data distribution in a table."""
        counts = labels.value_counts()
        # Use Histogram
        plt.figure(figsize=(10, 5))
        sns.histplot(labels, bins=20)
        plt.title("Data Distribution")
        plt.xlabel("Labels")
        plt.ylabel("Count")
        plt.show()
        # Pie
        labels_list = ["ham", "spam"]
        plt.figure(figsize=(10, 5))
        plt.pie(
            counts, labels=labels_list, autopct="%1.1f%%", colors=["lightblue", "blue"]
        )
        plt.title("Data Distribution")
        plt.show()

    def split_data(
        self,
    ) -> Dict[str, Tuple[SparseMatrix, NDArray[np.float64]]]:
        """Apply preprocessing and split data into training, validation, and test sets."""
        # Initalize Data
        data_series = pd.Series(self.data, name="text")
        labels_series = pd.Series(self.labels, name="label")
        # Report Data Distribution
        if self.report:
            self._report_data_distribution(data_series, labels_series)
        # Preprocess Data
        df = self._preprocess_data(
            pd.DataFrame({"text": data_series, "label": labels_series})
        )
        # Split into Train and Test Split.
        X_train, X_val, X_test, y_train, y_val, y_test = (
            self._manual_train_val_test_split(
                df["text_encoded"],
                df["label_encoded"],
                test_size=self.test_size,
                val_size=self.val_size,
            )
        )
        # Transform each into the right format for the model.
        train_df = pd.DataFrame(
            {"text_encoded": list(X_train), "label_encoded": y_train}
        )
        val_df = pd.DataFrame({"text_encoded": list(X_val), "label_encoded": y_val})
        test_df = pd.DataFrame({"text_encoded": list(X_test), "label_encoded": y_test})

        X_train_imm: SparseMatrix = csr_matrix(
            np.vstack(train_df["text_encoded"].values.tolist())
        )
        X_val_imm: SparseMatrix = csr_matrix(
            np.vstack(val_df["text_encoded"].values.tolist())
        )
        X_test_imm: SparseMatrix = csr_matrix(
            np.vstack(test_df["text_encoded"].values.tolist())
        )
        # Transform to 1d for later use.
        y_train: NDArray[np.float64] = train_df["label_encoded"].to_numpy().ravel()
        y_val: NDArray[np.float64] = val_df["label_encoded"].to_numpy().ravel()
        y_test: NDArray[np.float64] = test_df["label_encoded"].to_numpy().ravel()
        # Apply scaler
        X_train, X_val, X_test = self._apply_scaler(X_train_imm, X_val_imm, X_test_imm)
        # Return
        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    def _apply_scaler(
        self, X_train: SparseMatrix, X_val: SparseMatrix, X_test: SparseMatrix
    ) -> Tuple[SparseMatrix, SparseMatrix, SparseMatrix]:
        """Apply the scaler to the training, validation, and test sets."""
        self._scaler.partial_fit(X_train)

        def transform_sparse(X):
            """Apply the scaler to a sparse matrix."""
            X_scaled = self._scaler.transform(X)
            return X_scaled if not issparse(X_scaled) else X_scaled

        X_train_scaled = SparseMatrix(transform_sparse(X_train))
        X_val_scaled = SparseMatrix(transform_sparse(X_val))
        X_test_scaled = SparseMatrix(transform_sparse(X_test))
        return X_train_scaled, X_val_scaled, X_test_scaled

    def _manual_train_val_test_split(
        self, X, y, test_size=0.2, val_size=0.1
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Same as the lecture code, but also does val split"""
        num_data_points = X.shape[0]
        shuffled_indices = np.random.permutation(num_data_points)
        test_set_size = int(num_data_points * test_size)
        val_set_size = int(num_data_points * val_size)
        test_indices = shuffled_indices[:test_set_size]
        val_indices = shuffled_indices[test_set_size : test_set_size + val_set_size]
        train_indices = shuffled_indices[test_set_size + val_set_size :]

        X_train = X.iloc[train_indices]
        X_val = X.iloc[val_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_val = y.iloc[val_indices]
        y_test = y.iloc[test_indices]

        return (
            X_train.to_numpy(),
            X_val.to_numpy(),
            X_test.to_numpy(),
            y_train.to_numpy(),
            y_val.to_numpy(),
            y_test.to_numpy(),
        )

    def _preprocess_data(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> pd.DataFrame:
        """Apply text cleaning, tokenization, stemming, and TF-IDF."""
        processed_df = df.copy()
        # Tokenize and stem
        processed_df[text_column] = self._tokenize_and_stem(
            self._clean_text(processed_df[text_column])
        )
        # Encode labels
        processed_df["label_encoded"] = (
            processed_df["label"].astype("category").cat.codes
        )
        # Apply TF-IDF
        processed_df["text_encoded"] = list(
            self._fit_transform_data_tfidf(processed_df)
        )

        return processed_df

    def _clean_text(self, text: Union[str, pd.Series]) -> Union[str, pd.Series]:
        """Remove punctuation, URLs, and numbers, and convert text to lowercase."""
        if isinstance(text, pd.Series):
            return text.apply(self._clean_single_text)
        return self._clean_single_text(text)

    def _clean_single_text(self, text: str) -> str:
        """Clean a single text string."""
        text = text.strip()
        text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )  # Remove punctuation
        return text.lower()

    def _tokenize_and_stem(self, text: Union[str, pd.Series]) -> Union[str, pd.Series]:
        """Tokenize text, remove stopwords, and apply stemming."""
        if isinstance(text, pd.Series):
            return text.apply(self._tokenize_and_stem_single)
        return self._tokenize_and_stem_single(text)

    def _tokenize_and_stem_single(self, text: str) -> str:
        """Process a single text string."""
        tokens = word_tokenize(text)
        tokens = [
            self._stemmer.stem(word) for word in tokens if word not in self._stop_words
        ]
        return " ".join(tokens)

    def _fit_transform_data_tfidf(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> List[np.ndarray]:
        """Fit the TF-IDF vectorizer and transform text data using TF-IDF."""
        # Weird Type Errors /shrug
        return [
            np.array(arr)
            for arr in self._vectorizer.fit_transform(df[text_column].values).toarray()
        ]

    def preprocess_single_sentence(self, sentence: str) -> SparseMatrix:
        """Preprocess a single sentence."""
        cleaned_sentence = self._clean_single_text(sentence)
        tokenized_sentence = self._tokenize_and_stem_single(cleaned_sentence)
        tfidf_sentence = list(
            self._vectorizer.transform([tokenized_sentence]).toarray()
        )
        scaled_sentence = SparseMatrix(self._scaler.transform(tfidf_sentence))
        return scaled_sentence


class SMSSpamPreprocessor:
    """Loading data from the file and calling the TextPreprocessor to preprocess it."""

    def __init__(self, report: bool = False):
        self.filepath = "data/SMSSpamCollection"
        self.report = report

    def load_data_from_file(self, filepath: str) -> Tuple[List[str], List[str]]:
        """Load data from the file and return it as a list of sentences and labels."""
        labels = []
        sentences = []
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    label, sentence = parts
                    labels.append(label)
                    sentences.append(sentence)
        return sentences, labels

    def get_data_splits(
        self,
    ) -> Dict[str, Tuple[SparseMatrix, NDArray[np.float64]]]:
        """Load the data from the file and split it into training, validation, and test sets."""
        sentences, labels = self.load_data_from_file(self.filepath)
        print(
            f"📬 Loaded SMS Spam data 📬 from file. With \n {len(sentences)} sentences and \n {len(labels)} labels "
        )
        self.preprocessor = TextPreprocessor(
            data=sentences, labels=labels, report=self.report
        )
        # Save for later usage
        return self.preprocessor.split_data()

    def preprocess_single_sentence(self, sentence: str) -> SparseMatrix:
        """Preprocess a single sentence."""
        return self.preprocessor.preprocess_single_sentence(sentence)


class BookPreprocessor:
    """Loading data from the file and calling the TextPreprocessor to preprocess it."""

    def __init__(self, report: bool = False):
        self.filepath = "data/books.txt"
        self.report = report

    def load_data_from_file(self, filepath: str) -> Tuple[List[str], List[str]]:
        """Load data from the file and return it as a list of sentences and labels."""
        labels: list[Any] = []
        sentences: list[Any] = []
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    label, sentence = parts
                    labels.append(label)
                    sentences.append(sentence)

        return sentences, labels

    def get_data_splits(
        self,
    ) -> Dict[str, Tuple[SparseMatrix, NDArray[np.float64]]]:
        """Load the data from the file and split it into training, validation, and test sets."""
        sentences, labels = self.load_data_from_file(self.filepath)
        print(
            f"📬 Loaded Book data 📬 from file. With \n {len(sentences)} sentences and \n {len(labels)} labels "
        )
        print(f"Unique Labels: {set(labels)}")
        self.preprocessor = TextPreprocessor(
            data=sentences, labels=labels, scaler=StandardScaler(with_mean=False)
        )
        return self.preprocessor.split_data()

    def preprocess_single_sentence(self, sentence: str) -> SparseMatrix:
        """Preprocess a single sentence."""
        return self.preprocessor.preprocess_single_sentence(sentence)


if __name__ == "__main__":
    data = SMSSpamPreprocessor()
    split_data = data.get_data_splits()
