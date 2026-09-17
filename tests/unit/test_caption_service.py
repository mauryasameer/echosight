from unittest.mock import MagicMock

import tensorflow as tf

from src.models.decoder import RNN_Decoder
from src.models.encoder import CNN_Encoder
from src.services.caption_service import greedy_caption


class _FakeTokenizer:
    def __init__(self):
        self.word_index = {"<start>": 1, "<end>": 2, "<pad>": 0, "dog": 3, "runs": 4}
        self.index_word = {v: k for k, v in self.word_index.items()}


def test_greedy_caption_stops_at_end_token(monkeypatch):
    embedding_dim, units, vocab_size = 8, 16, 5
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    tokenizer = _FakeTokenizer()

    # force the decoder to always predict <end> (id 2) immediately
    def fake_decoder_call(x, features, hidden):
        batch = x.shape[0]
        logits = tf.one_hot([2] * batch, depth=vocab_size) * 100.0
        return logits, decoder.reset_state(batch), tf.ones((batch, 64, 1)) / 64.0

    monkeypatch.setattr(decoder, "call", fake_decoder_call)
    monkeypatch.setattr(
        "src.services.caption_service.load_and_preprocess_image", lambda path: tf.zeros((299, 299, 3))
    )

    fake_extractor = MagicMock(return_value=tf.random.normal((1, 8, 8, 2048)))

    result = greedy_caption(
        "/fake/img.jpg", encoder, decoder, fake_extractor, tokenizer, max_length=10
    )
    assert result.tokens[-1] == "<end>" or len(result.tokens) <= 10
    assert result.attention_plot.shape[1] == 64
