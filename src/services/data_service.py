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
    img_train, img_test, cap_train, cap_test = train_test_split(
        list(df["Path"]), cap_vector, test_size=test_size, random_state=42
    )
    return TrainTestSplit(img_train=img_train, img_test=img_test, cap_train=cap_train, cap_test=cap_test)
