import numpy as np
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


def test_dropout_is_inactive_by_default_and_active_when_training():
    """Regression test: the encoder's Dropout layer used to be constructed but never
    called (dead code) -- the real v0.1.0 training run had zero regularization. This
    checks the wiring both ways: calling without `training=True` (inference, the
    default) must be deterministic across repeated calls on the same input, while
    calling with `training=True` must actually inject randomness (dropout zeroing
    different units each call) rather than being a no-op."""
    encoder = CNN_Encoder(embedding_dim=256)
    features = tf.random.normal((2, 64, 2048))

    out_infer_1 = encoder(features, training=False).numpy()
    out_infer_2 = encoder(features, training=False).numpy()
    assert np.array_equal(out_infer_1, out_infer_2), "inference-mode output must be deterministic"

    out_default = encoder(features).numpy()
    assert np.array_equal(out_default, out_infer_1), "training defaults to False when omitted"

    tf.random.set_seed(0)
    out_train_1 = encoder(features, training=True).numpy()
    tf.random.set_seed(1)
    out_train_2 = encoder(features, training=True).numpy()
    assert not np.array_equal(out_train_1, out_train_2), (
        "training=True must actually apply dropout (stochastic across calls), not be a no-op"
    )
