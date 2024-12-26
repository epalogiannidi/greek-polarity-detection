import os
import re
import time
import unicodedata
from datetime import timedelta

import fasttext
import nltk
import numpy as np
import pandas as pd
import spacy
import torch
import yaml
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from tqdm import tqdm

from greek_polarity_detection import Logger

nltk.download("punkt")
logger = Logger().logger


class Timer:
    """Context manager to count elapsed time

    Example:
    --------
    .. highlight:: python
    .. code-block:: python

        with Timer() as t:
            y = f(x)
        print(f'Invocation of f took {t.elapsed}s!')
    """

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self._end = time.time()
        self._elapsed = self._end - self._start
        self.elapsed = str(timedelta(seconds=self._elapsed))


class SetUpConfig:
    """Loads up a configuration file

    Attributes
    ----------
    config_path: str
        The path to the configuration file
    config_values: Dict
        The values loaded from the configuration file
    """

    def __init__(self):
        self.config_path = "configs/config.yml"
        self.config_values = dict()

        with open(self.config_path, "r") as cf:
            self.config_values = yaml.safe_load(cf)


class Dataset:
    def __init__(self, config):
        self.name = config["dataset"]
        self.nlp = spacy.load("el_core_news_sm")
        self.stop_words = set(stopwords.words("greek"))
        self.feature = "review"
        self.label = "label"
        self.folds_num = 10
        self.folds = []
        self.vocabulary = []

        self.data = self.load()

    def load(self):
        # Load
        data = pd.read_excel(os.path.join("data", self.name))
        positive = data[data["label"] == "Positive"]
        negative = data[data["label"] == "Negative"]
        logger.info(
            f"Loaded dataset of {len(data)} rows: {len(positive)} positive and {len(negative)} negative."
        )

        # preprocess
        data["processed"] = data["review"].progress_apply(
            lambda x: self.preprocess(x.strip())
        )

        self.vocabulary = list(set(self.vocabulary))

        # create folds
        self.split_train_test(data)
        return data

    def split_train_test(self, data):
        X = data[[self.feature, "processed"]]  # Features as DataFrame
        y = data[self.label]
        skf = StratifiedKFold(n_splits=self.folds_num, shuffle=False, random_state=None)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.folds.append(
                (
                    {
                        "train_data": X_train,
                        "train_labels": y_train,
                        "test_data": X_test,
                        "test_labels": y_test,
                    }
                )
            )

    def preprocess(self, text: str):

        text = re.sub(r"[^\w\s]", "", text).lower()
        doc = self.nlp(text)

        tokens = []
        postags = []

        for token in doc:
            if token.text not in self.stop_words:
                tokens.append(token.text)
                postags.append(token.pos_)
        content_words = [
            tokens[i]
            for i, postag in enumerate(postags)
            if postag in ["ADV", "ADJ", "VERB", "NOUN"] and not tokens[i].isdigit()
        ]
        self.vocabulary.extend(content_words)
        return content_words


class Lexicon:
    def __init__(self, config):
        self.name = config["lexicon"]
        self.dim = "val"
        self.lexicon = pd.DataFrame(
            {
                "words": self.load(
                    os.path.join("data", self.name, "anew_greek.txt"), False
                ),
                self.dim: self.load(os.path.join("data", self.name, f"{self.dim}.txt")),
            }
        )
        self.seeds = 600
        self.seed_lexicon = self.create_seed_lexicon()
        self.seed_lexicon_words = list(self.seed_lexicon.keys())
        self.coefs = {"val": [], "aro": [], "dom": []}

    def load(self, lex: str, to_float: bool = True):
        with open(lex, "r") as f:
            lines = f.readlines()
        return (
            [float(l.strip()) for l in lines]
            if to_float
            else [l.strip() for l in lines]
        )

    def create_seed_lexicon(self):
        sorted_df = self.lexicon.sort_values(
            by=self.dim, ascending=False
        )  # Use ascending=False for descending order
        lowest = sorted_df.tail(self.seeds // 2)  # N rows with the lowest values
        highest = sorted_df.head(self.seeds // 2)  # N rows with the highest values

        concatenated = pd.concat([lowest, highest])
        sconcatenated = concatenated.sort_values(by="words")
        return sconcatenated.set_index("words")[self.dim].to_dict()


def strip_accents_and_lowercase(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    ).lower()


class LexiconTrainer:
    def __init__(self):
        self.lexicon = Lexicon(SetUpConfig().config_values)

        self.use_lse = True
        self.model = SentenceTransformer("lighteternal/stsb-xlm-r-greek-transfer")
        fasttext.util.download_model("el", if_exists="ignore")
        self.ft = fasttext.load_model("cc.el.300.bin")
        self.similarity_matrix = self.compute_similarity_matrix(
            self.lexicon.seed_lexicon_words, self.lexicon.seed_lexicon_words
        )

    def get_word_embedding_st(self, word):
        return self.model.encode(word, convert_to_tensor=True)

    # Function to get embeddings for words
    def get_word_embedding(self, word):

        inputs = self.tokenizer(
            word, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # We take the mean of the token embeddings for the word (this gives a single vector per word)
        # embeddings = outputs.last_hidden_state.mean(dim=1)
        token_embeddings = outputs.last_hidden_state.squeeze(
            0
        )  # Removing the batch dimension

        # Get the tokens (subwords)
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze().tolist()
        )

        # Extract embeddings corresponding to the word itself (excluding [CLS] and [SEP])
        word_token_embeddings = token_embeddings[
            1:-1
        ]  # Exclude [CLS] (index 0) and [SEP] (index -1)

        # If there are multiple subwords, average them
        word_embedding = torch.mean(
            word_token_embeddings, dim=0
        )  # Averaging the subword embeddings

        return word_embedding

    def compute_similarity_matrix(self, lvector, rvector):
        # FASTTEXT
        lembeddings = np.array(
            [self.ft.get_word_vector(word) for word in lvector]
        )  # Shape: (N, D)
        rembeddings = np.array(
            [self.ft.get_word_vector(word) for word in rvector]
        )  # Shape: (N, D)

        lembeddings_tensor = torch.tensor(lembeddings).cpu()
        rembeddings_tensor = torch.tensor(rembeddings).cpu()

        # TRANSFORMERS
        lembeddingsft = [self.get_word_embedding_st(word) for word in tqdm(lvector)]
        rembeddingsft = [self.get_word_embedding_st(word) for word in tqdm(rvector)]

        # Convert list of embeddings to a tensor
        lembeddings_tensorft = torch.stack(lembeddingsft).cpu()
        rembeddings_tensorft = torch.stack(rembeddingsft).cpu()

        pca = PCA(n_components=300)
        ldecomp = pca.fit_transform(lembeddings_tensorft)
        rdecomp = pca.fit_transform(rembeddings_tensorft)

        lembeddings_concat = (lembeddings_tensor + ldecomp) / 2
        rembeddings_concat = (rembeddings_tensor + rdecomp) / 2

        normalized_lembeddings = torch.nn.functional.normalize(
            lembeddings_concat, dim=1
        )
        normalized_rembeddings = torch.nn.functional.normalize(
            rembeddings_concat, dim=1
        )

        similarity_matrix = torch.mm(normalized_lembeddings, normalized_rembeddings.T)

        return similarity_matrix

    # Function to compute similarity matrix for N words
    def compute_similarity_matrix_(self, lvector, rvector):
        lembeddings = np.array(
            [self.ft.get_word_vector(word) for word in lvector]
        )  # Shape: (N, D)
        rembeddings = np.array(
            [self.ft.get_word_vector(word) for word in rvector]
        )  # Shape: (N, D)

        lembeddings_tensor = torch.tensor(lembeddings)
        rembeddings_tensor = torch.tensor(rembeddings)

        normalized_lembeddings = torch.nn.functional.normalize(
            lembeddings_tensor, dim=1
        )
        normalized_rembeddings = torch.nn.functional.normalize(
            rembeddings_tensor, dim=1
        )
        similarity_matrix = torch.mm(normalized_lembeddings, normalized_rembeddings.T)

        return similarity_matrix

    def tune_lambda(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        ridge = Ridge()

        # Define the parameter grid for lambda (alpha in Ridge)
        param_grid = {
            "alpha": np.logspace(
                -10, 10, 13
            )  # Example grid: values for alpha from 1e-6 to 1e6
        }

        # Initialize GridSearchCV with 5-fold cross-validation
        grid_search = GridSearchCV(
            estimator=ridge,
            param_grid=param_grid,
            cv=10,
            scoring="neg_mean_squared_error",
            verbose=1,
        )

        # Fit the model to the training data
        grid_search.fit(X_train, y_train)

        # Print the best parameter (lambda) and best score
        print(f"Best alpha (lambda): {grid_search.best_params_['alpha']}")
        print(f"Best cross-validation score: {grid_search.best_score_}")

        # Evaluate the best model on the test set
        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        print(f"Test set score: {test_score}")
        return grid_search.best_params_["alpha"]

    def lse(self, emotion_ratings, X):
        return np.linalg.lstsq(X, emotion_ratings, rcond=None)[0]

    def rr(self, emotion_ratings, X):
        lambda_reg = self.tune_lambda(X, emotion_ratings)
        # lambda_reg = 1
        ridge = Ridge(
            alpha=lambda_reg, fit_intercept=False
        )  # fit_intercept=False because we manually added bias
        ridge.fit(X, emotion_ratings)
        return ridge.coef_

    def model_training(self, dim):
        emotion_ratings = np.array(list(self.lexicon.seed_lexicon.values()))
        self.similarity_matrix = self.similarity_matrix.cpu()
        sim_val_matrix = self.similarity_matrix * emotion_ratings[:, np.newaxis]

        X = np.hstack([np.ones((sim_val_matrix.shape[0], 1)), sim_val_matrix])

        if self.use_lse:
            self.lexicon.coefs["val"] = self.lse(emotion_ratings, X)
        else:
            self.lexicon.coefs["val"] = self.rr(emotion_ratings, X)
