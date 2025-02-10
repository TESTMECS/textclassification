import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from scipy.sparse import csr_matrix

# Typing
from numpy._typing._array_like import NDArray
from typing import Any, Tuple, Dict, Union, List
from numpy._typing import NDArray

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Set random seed for reproducibility
np.random.seed(seed=420)

# Define type alias for sparse matrix
SparseMatrix = csr_matrix


class TextPreprocessor:
    def __init__(
        self,
        data: List[str],
        labels: List[str],
        test_size: float = 0.2,
        val_size: float = 0.1,
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
        self._scaler = StandardScaler()

    def split_data(self) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Apply preprocessing and split data into training, validation, and test sets."""
        # Initalize Data
        data_series = pd.Series(self.data, name="text")
        labels_series = pd.Series(self.labels, name="label")
        # Preprocess Data
        df = self._preprocess_data(
            pd.DataFrame({"text": data_series, "label": labels_series})
        )
        # Split
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

        X_train_imm: NDArray[np.float64] = np.vstack(
            train_df["text_encoded"].values.tolist()
        )
        X_val_imm: NDArray[np.float64] = np.vstack(
            val_df["text_encoded"].values.tolist()
        )
        X_test_imm: NDArray[np.float64] = np.vstack(
            test_df["text_encoded"].values.tolist()
        )
        # Transform to 1d for later use.
        y_train: NDArray[np.float64] = train_df["label_encoded"].to_numpy().ravel()
        y_val: NDArray[np.float64] = val_df["label_encoded"].to_numpy().ravel()
        y_test: NDArray[np.float64] = test_df["label_encoded"].to_numpy().ravel()
        # Final Transformation: Standard Scaling of text data
        X_train: NDArray[np.float64] = np.array(
            self._scaler.fit_transform(X_train_imm)
        ).astype(np.float64)
        X_test: NDArray[np.float64] = np.array(
            self._scaler.transform(X_test_imm)
        ).astype(np.float64)
        X_val: NDArray[np.float64] = np.array(self._scaler.transform(X_val_imm)).astype(
            np.float64
        )
        # Return
        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

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
        processed_df[text_column] = self._tokenize_and_stem(
            self._clean_text(processed_df[text_column])
        )
        processed_df["label_encoded"] = processed_df["label"].apply(
            lambda x: 1 if x == "spam" else 0
        )
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

    def preprocess_single_sentence(self, sentence: str) -> NDArray[np.float64]:
        """Preprocess a single sentence."""
        cleaned_sentence = self._clean_single_text(sentence)
        tokenized_sentence = self._tokenize_and_stem_single(cleaned_sentence)
        tfidf_sentence = list(
            self._vectorizer.transform([tokenized_sentence]).toarray()
        )
        scaled_sentence = np.array(self._scaler.transform(tfidf_sentence)).astype(
            np.float64
        )
        return scaled_sentence


class SMSSpamPreprocessor:
    """Loading data from the file and calling the TextPreprocessor to preprocess it."""

    def __init__(self, filepath: str):
        self.filepath = filepath

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
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Load the data from the file and split it into training, validation, and test sets."""
        sentences, labels = self.load_data_from_file(self.filepath)
        preprocessor = TextPreprocessor(data=sentences, labels=labels)
        # Save for later usage
        self._vectorizer = preprocessor._vectorizer
        self._scaler = preprocessor._scaler
        return preprocessor.split_data()


if __name__ == "__main__":
    # Example Usage
    data = SMSSpamPreprocessor("data/SMSSpamCollection")
    split_data = data.get_data_splits()
    # Print the shape of the training set
    print(split_data["train"][0].shape)
