from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CNN_Encoder(keras.Model):
    """Projects cached InceptionV3 features (batch, 64, 2048) into embedding_dim."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.fc = layers.Dense(embedding_dim)
        self.dropout = layers.Dropout(0.5)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
