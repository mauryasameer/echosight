import tensorflow as tf

from src.models.encoder import CNN_Encoder


def test_encoder_output_shape():
    encoder = CNN_Encoder(embedding_dim=256)
    features = tf.random.normal((2, 64, 2048))
    output = encoder(features)
    assert output.shape == (2, 64, 256)


def test_encoder_output_is_nonnegative_after_relu():
    encoder = CNN_Encoder(embedding_dim=256)
    features = tf.random.normal((2, 64, 2048))
    output = encoder(features)
    assert tf.reduce_min(output).numpy() >= 0.0
