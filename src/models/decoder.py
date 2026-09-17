from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.attention import Attention_model


class RNN_Decoder(keras.Model):
    """GRU decoder with Bahdanau attention, one token per call (teacher-forced during training)."""

    def __init__(self, embedding_dim: int, units: int, vocab_size: int) -> None:
        super().__init__()
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(
            self.units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform"
        )
        self.fc1 = layers.Dense(self.units)
        self.fc2 = layers.Dense(vocab_size)
        self.attention = Attention_model(self.units)

    def call(
        self, x: tf.Tensor, features: tf.Tensor, hidden: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size: int) -> tf.Tensor:
        return tf.zeros((batch_size, self.units))
