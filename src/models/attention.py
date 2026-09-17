from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Attention_model(keras.Model):
    """Bahdanau (additive) attention over the encoder's 64 spatial regions."""

    def __init__(self, units: int) -> None:
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
        self.units = units

    def call(self, features: tf.Tensor, hidden: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # features: (batch, 64, embedding_dim); hidden: (batch, units)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
