import tensorflow as tf

from src.models.decoder import RNN_Decoder


def test_decoder_forward_pass_shapes():
    decoder = RNN_Decoder(embedding_dim=256, units=512, vocab_size=5001)
    features = tf.random.normal((2, 64, 256))
    hidden = decoder.reset_state(batch_size=2)
    dec_input = tf.expand_dims([1, 1], 1)
    predictions, state, attention_weights = decoder(dec_input, features, hidden)
    assert predictions.shape == (2, 5001)
    assert state.shape == (2, 512)
    assert attention_weights.shape == (2, 64, 1)


def test_reset_state_shape():
    decoder = RNN_Decoder(embedding_dim=256, units=512, vocab_size=5001)
    hidden = decoder.reset_state(batch_size=3)
    assert hidden.shape == (3, 512)
