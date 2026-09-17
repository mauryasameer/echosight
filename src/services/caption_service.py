from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.decoder import RNN_Decoder
from src.models.encoder import CNN_Encoder
from src.services.feature_service import load_and_preprocess_image


@dataclass
class CaptionResult:
    tokens: list[str]
    attention_plot: np.ndarray


def greedy_caption(
    image_path: str,
    encoder: CNN_Encoder,
    decoder: RNN_Decoder,
    extractor: keras.Model,
    tokenizer,
    max_length: int,
) -> CaptionResult:
    attention_plot = np.zeros((max_length, 64))
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_and_preprocess_image(image_path), 0)
    img_tensor_val = extractor(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]], 0)
    result: list[str] = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = int(tf.argmax(predictions[0]).numpy())
        word = tokenizer.index_word.get(predicted_id, "<unk>")
        result.append(word)

        if word == "<end>":
            break
        dec_input = tf.expand_dims([predicted_id], 0)

    return CaptionResult(tokens=result, attention_plot=attention_plot[: len(result)])


def beam_caption(
    image_path: str,
    encoder: CNN_Encoder,
    decoder: RNN_Decoder,
    extractor: keras.Model,
    tokenizer,
    max_length: int,
    beam_index: int = 3,
) -> CaptionResult:
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_and_preprocess_image(image_path), 0)
    img_tensor_val = extractor(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)

    start = [tokenizer.word_index["<start>"]]
    sequences: list[list] = [[start, 0.0]]
    end_id = tokenizer.word_index.get("<end>")

    for _ in range(max_length - 1):
        all_candidates = []
        for seq, score in sequences:
            if end_id is not None and seq[-1] == end_id:
                all_candidates.append([seq, score])
                continue

            dec_input = tf.expand_dims([seq[-1]], 0)
            predictions, _, _ = decoder(dec_input, features, hidden)
            log_probs = tf.nn.log_softmax(predictions[0]).numpy()
            top_ids = np.argsort(log_probs)[-beam_index:]
            for token_id in top_ids:
                candidate = [seq + [int(token_id)], score + float(log_probs[token_id])]
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda entry: entry[1], reverse=True)
        sequences = ordered[:beam_index]

        if end_id is not None and all(seq[-1] == end_id for seq, _ in sequences):
            break

    best_seq = sequences[0][0]
    result = [tokenizer.index_word.get(token_id, "<unk>") for token_id in best_seq[1:]]

    attention_plot = np.zeros((max(len(result), 1), 64))
    return CaptionResult(tokens=result, attention_plot=attention_plot[: len(result)])
