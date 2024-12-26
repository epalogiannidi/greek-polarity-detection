import os
import random
from itertools import chain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from tqdm import tqdm

from greek_polarity_detection import Dataset, LexiconTrainer, SetUpConfig, logger
from greek_polarity_detection.utils import Timer

tqdm.pandas()
import torch
from tqdm import tqdm


def llm_setup():
    llm = ChatOllama(model="aya-expanse:8b-q4_K_M", temperature=0.5)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a polarity classification assistant. Classify reviews written in Greek as either 'Positive' or 'Negative'. Respond with just one word: 'Positive' or 'Negative'. Do not include explanations, punctuation, or any additional text.",
            ),
            (
                "human",
                "Review: {review}\nProvide your answer as a single word, either 'Positive' or 'Negative':\n",
            ),
        ]
    )

    chainl = prompt | llm

    return chainl

def compute_score(words, d_instance, sim_mat_coefs, lexicon, min_v_score, max_v_score):
    v_scores = []
    for w in words:
        if w not in d_instance.vocabulary:
            continue
        w_index = d_instance.vocabulary.index(w)
        v_score = (torch.sum(sim_mat_coefs[w_index])).item()
        score = v_score
        if min_v_score is not None:
            scaled_score = float(
                2 * ((v_score - min_v_score) / (max_v_score - min_v_score)) - 1
            )
            score = scaled_score
        v_scores.append(score)
    if v_scores == []:
        v_scores = [0]
    return v_scores


def max_counts(scores):
    pos = 0
    neg = 0
    for w in scores:
        if w > 0:
            pos += 1
        else:
            neg += 1
    return "Positive" if pos > neg else "Negative"


def average_score(scores):
    score = ""
    if len(scores) == 0:
        score = "Negative"
    else:
        score = "Positive" if np.array(scores).mean() > 0 else "Negative"
    return score


def polarity_ratio(scores):
    pos = 0
    neg = 0
    for w in scores:
        if w > 0:
            pos += 1
        else:
            neg += 1
    return "Positive" if pos - neg > 0 else "Negative"


def max_dim_score(scores):
    maxs = np.array(scores).max()
    mins = np.array(scores).min()
    return "Positive" if maxs - abs(mins) > 0 else "Negative"


def strong_negative(scores):

    hp = [s for s in scores if s >= 0.4]
    hn = [s for s in scores if s <= -0.4]
    normal = [s for s in scores if -0.4 < s < 0.4]

    if len(hp) > len(hn):
        score = "Positive"
    elif len(hn) > len(hp):
        score = "Negative"
    else:
        score = average_score(normal)
    return score


def weighted_average(scores):
    # higher weight at the beginning
    weights = np.linspace(1, 0.1, num=len(scores))
    weights /= weights.sum()
    return (
        "Positive" if np.dot(scores, weights) > 0 else "Negative"
    )  # Dot product of values and weights


def majority_vote(row):
    return row.mode()[0]  # Get the most frequent value in the row


def execute():

    config = SetUpConfig().config_values
    dataset_instance = Dataset(config)
    lexicon_trainer = LexiconTrainer()

    lexicon = lexicon_trainer.lexicon
    lexicon_trainer.model_training(dim="val")

    sim_mat = lexicon_trainer.compute_similarity_matrix(
        dataset_instance.vocabulary, lexicon.seed_lexicon_words
    )

    sim_mat = sim_mat.cpu()
    sim_mat_bias = np.hstack([np.ones((sim_mat.shape[0], 1)), sim_mat])

    sim_mat_coefs = torch.tensor(sim_mat_bias) * torch.tensor(lexicon.coefs["val"])
    for fold in tqdm(dataset_instance.folds, position=0):
        fold["time"] = {}
        fold["train_data"]["val"] = fold["train_data"]["processed"].progress_apply(
            lambda x: compute_score(
                x, dataset_instance, sim_mat_coefs, lexicon, None, None
            )
        )
        # scaling based on the training
        min_score = np.array(list(chain.from_iterable(fold["train_data"]["val"]))).min()
        max_score = np.array(list(chain.from_iterable(fold["train_data"]["val"]))).max()
        with Timer() as tlex:
            fold["test_data"]["val"] = fold["test_data"]["processed"].progress_apply(
                lambda x: compute_score(
                    x, dataset_instance, sim_mat_coefs, lexicon, min_score, max_score
                )
            )
            fold["test_data"]["label_average"] = fold["test_data"][
                "val"
            ].progress_apply(lambda x: average_score(x))
            fold["test_data"]["label_waverage"] = fold["test_data"][
                "val"
            ].progress_apply(lambda x: weighted_average(x))
            fold["test_data"]["label_max_counts"] = fold["test_data"][
                "val"
            ].progress_apply(lambda x: max_counts(x))
            fold["test_data"]["label_max"] = fold["test_data"]["val"].progress_apply(
                lambda x: max_dim_score(x)
            )
            fold["test_data"]["label_polarity_ratio"] = fold["test_data"][
                "val"
            ].progress_apply(lambda x: polarity_ratio(x))
            fold["test_data"]["label_strong_negative"] = fold["test_data"][
                "val"
            ].progress_apply(lambda x: strong_negative(x))
            fold["test_data"]["label_random"] = fold["test_data"]["val"].progress_apply(
                lambda x: random.choice(["Positive", "Negative"])
            )
        fold["time"]["tlex"] = tlex.elapsed

        with Timer() as tllm:
            chainl = llm_setup()
            fold["test_data"]["llm_polarity"] = [
                chainl.invoke(
                    {
                        "review": review,
                        "words": fold["test_data"]["processed"].iloc[idx],
                        "ratings": fold["test_data"]["val"].iloc[idx],
                    },
                ).content.strip()
                for idx, review in tqdm(enumerate(fold["test_data"]["review"]))
            ]
        fold["time"]["llm"] = tllm.elapsed
        fold["test_data"]["label_majority"] = fold["test_data"][
            ["label_average", "llm_polarity"]
        ].apply(majority_vote, axis=1)

        # Keep the words only once, remove the most common words from the vocabulary
        fold["accuracy"] = {}
        fold["accuracy"]["llm"] = sum(
            t == p
            for t, p in zip(
                list(fold["test_labels"]), list(fold["test_data"]["llm_polarity"])
            )
        ) / len(list(fold["test_labels"]))
        fold["accuracy"]["average"] = sum(
            t == p
            for t, p in zip(
                list(fold["test_labels"]), list(fold["test_data"]["label_average"])
            )
        ) / len(list(fold["test_labels"]))
        fold["accuracy"]["waverage"] = sum(
            t == p
            for t, p in zip(
                list(fold["test_labels"]), list(fold["test_data"]["label_waverage"])
            )
        ) / len(list(fold["test_labels"]))
        fold["accuracy"]["strong_negative"] = sum(
            t == p
            for t, p in zip(
                list(fold["test_labels"]),
                list(fold["test_data"]["label_strong_negative"]),
            )
        ) / len(list(fold["test_labels"]))
        fold["accuracy"]["max_counts"] = sum(
            t == p
            for t, p in zip(
                list(fold["test_labels"]), list(fold["test_data"]["label_max_counts"])
            )
        ) / len(list(fold["test_labels"]))
        fold["accuracy"]["polarity_ratio"] = sum(
            t == p
            for t, p in zip(
                list(fold["test_labels"]),
                list(fold["test_data"]["label_polarity_ratio"]),
            )
        ) / len(list(fold["test_labels"]))
        fold["accuracy"]["max"] = sum(
            t == p
            for t, p in zip(
                list(fold["test_labels"]), list(fold["test_data"]["label_max"])
            )
        ) / len(list(fold["test_labels"]))
        fold["accuracy"]["random"] = sum(
            t == p
            for t, p in zip(
                list(fold["test_labels"]), list(fold["test_data"]["label_random"])
            )
        ) / len(list(fold["test_labels"]))

        print("FIRST FOLD COMPLE")
        path = "results_comb"
        os.makedirs(path, exist_ok=True)
        for i, f in enumerate(dataset_instance.folds):
            os.makedirs(f"{path}/{i}", exist_ok=True)
            f["test_data"].to_csv(f"{path}/{i}/test.csv")
            f["test_labels"].to_csv(f"{path}/{i}/labels.csv")

            with open(f"{path}/{i}/accuracies.txt", "w") as file:
                for key, value in f["accuracy"].items():
                    file.write(f"{key}: {value}\n")
            with open(f"{path}/{i}/times.txt", "w") as file:
                for key, value in f["time"].items():
                    file.write(f"{key}: {value}\n")

    # Compute accuracy

    acc_average = np.array(
        [f["accuracy"]["average"] for f in dataset_instance.folds]
    ).mean()
    acc_max_counts = np.array(
        [f["accuracy"]["max_counts"] for f in dataset_instance.folds]
    ).mean()
    acc_polarity_ratio = np.array(
        [f["accuracy"]["polarity_ratio"] for f in dataset_instance.folds]
    ).mean()
    acc_random = np.array(
        [f["accuracy"]["random"] for f in dataset_instance.folds]
    ).mean()
    # acc_majority = np.array([f["accuracy"]["majority"] for f in dataset_instance.folds]).mean()
    acc_waverage = np.array(
        [f["accuracy"]["waverage"] for f in dataset_instance.folds]
    ).mean()
    acc_strong_negative = np.array(
        [f["accuracy"]["strong_negative"] for f in dataset_instance.folds]
    ).mean()
    acc_llm = np.array([f["accuracy"]["llm"] for f in dataset_instance.folds]).mean()

    logger.info(f"Cross Validation Accuracies: \n {acc_average=}\n{acc_max_counts=}\n{acc_polarity_ratio}\n{acc_random}\n{acc_waverage}\n{acc_strong_negative}\n{acc_llm}")

    path = "results_comb"
    os.makedirs(path, exist_ok=True)
    for i, f in enumerate(dataset_instance.folds):
        os.makedirs(f"{path}/{i}", exist_ok=True)
        f["test_data"].to_csv(f"{path}/{i}/test.csv")
        f["test_labels"].to_csv(f"{path}/{i}/labels.csv")

        with open(f"{path}/{i}/accuracies.txt", "w") as file:
            for key, value in f["accuracy"].items():
                file.write(f"{key}: {value}\n")
        with open(f"{path}/{i}/times.txt", "w") as file:
            for key, value in f["time"].items():
                file.write(f"{key}: {value}\n")
