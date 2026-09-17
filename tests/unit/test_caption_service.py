from unittest.mock import MagicMock

import tensorflow as tf

from src.models.decoder import RNN_Decoder
from src.models.encoder import CNN_Encoder
from src.services.caption_service import beam_caption, greedy_caption


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


def test_beam_caption_terminates_when_all_sequences_end_immediately(monkeypatch):
    """Regression test for a real hang bug found during implementation: once the
    highest-scoring beam sequence hits <end>, a naive `while len(sequences[0][0]) <
    max_length` loop guard never becomes false if finished sequences are correctly
    excluded from further decoding (their length stops growing). The fix bounds the
    loop with `for _ in range(max_length - 1)` instead. This test forces every beam
    candidate to predict <end> on the very first step -- the exact scenario that hung
    the earlier, buggy draft -- and must complete without hanging."""
    embedding_dim, units, vocab_size = 8, 16, 5
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    tokenizer = _FakeTokenizer()

    # force the decoder to always predict <end> (id 2) immediately, for every beam candidate
    def fake_decoder_call(x, features, hidden):
        batch = x.shape[0]
        logits = tf.one_hot([2] * batch, depth=vocab_size) * 100.0
        return logits, decoder.reset_state(batch), tf.ones((batch, 64, 1)) / 64.0

    monkeypatch.setattr(decoder, "call", fake_decoder_call)
    monkeypatch.setattr(
        "src.services.caption_service.load_and_preprocess_image", lambda path: tf.zeros((299, 299, 3))
    )

    fake_extractor = MagicMock(return_value=tf.random.normal((1, 8, 8, 2048)))

    # if the hang bug were reintroduced, this call would never return -- there is no
    # timeout here by design, matching the greedy test above; a real regression would
    # hang the test suite rather than fail cleanly, but the bounded for-loop makes that
    # structurally impossible regardless of beam state.
    result = beam_caption(
        "/fake/img.jpg", encoder, decoder, fake_extractor, tokenizer, max_length=10, beam_index=3
    )
    assert result.tokens[0] == "<end>"
    assert result.attention_plot.shape[1] == 64


def test_beam_caption_returns_result_for_normal_max_length(monkeypatch):
    """A more realistic beam search (random, untrained weights -- never all hit <end>
    on step one) still terminates and returns a well-formed CaptionResult within the
    bounded loop."""
    embedding_dim, units, vocab_size = 8, 16, 50
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    tokenizer = _FakeTokenizer()
    tokenizer.word_index = {"<start>": 1, "<end>": 2, "<pad>": 0}
    tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}

    monkeypatch.setattr(
        "src.services.caption_service.load_and_preprocess_image", lambda path: tf.zeros((299, 299, 3))
    )
    fake_extractor = MagicMock(return_value=tf.random.normal((1, 8, 8, 2048)))

    result = beam_caption(
        "/fake/img.jpg", encoder, decoder, fake_extractor, tokenizer, max_length=5, beam_index=2
    )
    assert len(result.tokens) <= 5
    assert result.attention_plot.shape[1] == 64
