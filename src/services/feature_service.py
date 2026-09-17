from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input


def build_feature_extractor() -> keras.Model:
    """Frozen InceptionV3 (ImageNet weights), output is the last conv layer's (8, 8, 2048) map."""
    image_model = keras.applications.InceptionV3(include_top=False, weights="imagenet")
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    return keras.Model(new_input, hidden_layer)


def load_and_preprocess_image(image_path: str) -> tf.Tensor:
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    return preprocess_input(img)


def _cache_key(image_path: str) -> str:
    return hashlib.sha256(image_path.encode("utf-8")).hexdigest()


def extract_and_cache_features(
    image_paths: list[str], extractor: keras.Model, cache_dir: str
) -> dict[str, np.ndarray]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    features: dict[str, np.ndarray] = {}

    for image_path in image_paths:
        key = _cache_key(image_path)
        cache_file = cache_path / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                features[image_path] = pickle.load(f)
            continue

        img = load_and_preprocess_image(image_path)
        batch = tf.expand_dims(img, 0)
        raw = extractor(batch)
        reshaped = tf.reshape(raw, (raw.shape[1] * raw.shape[2], raw.shape[3])).numpy()

        with open(cache_file, "wb") as f:
            pickle.dump(reshaped, f)
        features[image_path] = reshaped

    return features
