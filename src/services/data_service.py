from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def load_captions(captions_csv: str, images_dir: str) -> pd.DataFrame:
    all_img_id: list[str] = []
    all_img_path: list[str] = []
    annotations: list[str] = []
    with open(captions_csv) as f:
        for line in f.readlines()[1:]:
            img_id, cap = line.split(",", 1)
            img_id = img_id.split(".")[0]
            all_img_id.append(img_id)
            all_img_path.append(f"{images_dir}/{img_id}.jpg")
            annotations.append(cap.split("\n")[0])
    return pd.DataFrame({"ID": all_img_id, "Path": all_img_path, "Captions": annotations})


def build_annotations(df: pd.DataFrame) -> list[str]:
    return [f"<start> {cap} <end>" for cap in df["Captions"]]


def build_tokenizer(annotations: list[str], top_k: int = 5000) -> Tokenizer:
    tokenizer = Tokenizer(
        num_words=top_k, oov_token="<unk>", filters='!"#$%&()*+.-/:;=?@[\\]^_`{|}~ '
    )
    tokenizer.fit_on_texts(annotations)
    tokenizer.word_index["<pad>"] = 0
    tokenizer.index_word[0] = "<pad>"
    return tokenizer


def texts_to_padded_sequences(tokenizer: Tokenizer, annotations: list[str]) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(annotations)
    max_length = max(len(t) for t in sequences)
    return pad_sequences(sequences, padding="post", maxlen=max_length)


@dataclass
class TrainTestSplit:
    img_train: list[str]
    img_test: list[str]
    cap_train: np.ndarray
    cap_test: np.ndarray


def split_train_test(df: pd.DataFrame, cap_vector: np.ndarray, test_size: float = 0.2) -> TrainTestSplit:
    """Splits by IMAGE, not by caption row. `df` has one row per caption (5 per image
    in Flickr8k); splitting rows directly (as an earlier version of this function did)
    lets the same image's captions land on both sides of the split, leaking nearly
    every image into both train and test. Grouping by `df["ID"]` first keeps all of an
    image's captions together on one side."""
    unique_ids = np.array(sorted(df["ID"].unique().tolist()), dtype=object)
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=42)
    train_mask = df["ID"].isin(set(train_ids)).to_numpy()
    test_mask = df["ID"].isin(set(test_ids)).to_numpy()

    img_train = list(df.loc[train_mask, "Path"])
    img_test = list(df.loc[test_mask, "Path"])
    cap_train = cap_vector[train_mask]
    cap_test = cap_vector[test_mask]
    return TrainTestSplit(img_train=img_train, img_test=img_test, cap_train=cap_train, cap_test=cap_test)
