import tensorflow as tf

from src.models.attention import Attention_model


def test_attention_output_shapes():
    attention = Attention_model(units=512)
    features = tf.random.normal((2, 64, 256))
    hidden = tf.random.normal((2, 512))
    context_vector, attention_weights = attention(features, hidden)
    assert context_vector.shape == (2, 256)
    assert attention_weights.shape == (2, 64, 1)


def test_attention_weights_sum_to_one_per_example():
    attention = Attention_model(units=512)
    features = tf.random.normal((2, 64, 256))
    hidden = tf.random.normal((2, 512))
    _, attention_weights = attention(features, hidden)
    sums = tf.reduce_sum(attention_weights, axis=1)
    for s in sums.numpy().flatten():
        assert abs(s - 1.0) < 1e-4
