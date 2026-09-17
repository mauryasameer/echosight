from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.models.decoder import RNN_Decoder
from src.models.encoder import CNN_Encoder

logger = logging.getLogger(__name__)

_loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def _masked_loss(real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = _loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


class CaptionTrainer:
    """Orchestrates encoder/decoder training with teacher forcing and checkpointing."""

    def __init__(self, embedding_dim: int, units: int, vocab_size: int, checkpoint_dir: str) -> None:
        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)
        self.optimizer = keras.optimizers.Adam()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint = tf.train.Checkpoint(
            encoder=self.encoder,
            decoder=self.decoder,
            optimizer=self.optimizer,
            epoch=tf.Variable(0),
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, str(self.checkpoint_dir), max_to_keep=5
        )
        self._restore_latest()

    def _restore_latest(self) -> None:
        if self.checkpoint_manager.latest_checkpoint:
            try:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                logger.info("Resumed from checkpoint at epoch %d", int(self.checkpoint.epoch))
            except Exception:
                self.checkpoint.epoch.assign(0)
                logger.warning("Checkpoint restore failed, starting fresh", exc_info=True)
        else:
            logger.info("No checkpoint found, starting fresh")

    def train_step(self, img_tensor: tf.Tensor, target: tf.Tensor, start_token_id: int) -> tf.Tensor:
        loss = 0.0
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([start_token_id] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)
            for i in range(1, target.shape[1]):
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += _masked_loss(target[:, i], predictions)
                dec_input = tf.expand_dims(target[:, i], 1)

        avg_loss = loss / int(target.shape[1])
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables, strict=True))
        return avg_loss

    def fit(
        self,
        dataset: Iterable[tuple[tf.Tensor, tf.Tensor]],
        epochs: int,
        start_token_id: int,
        checkpoint_interval: int = 1,
        on_epoch_end: Callable[[int, float], None] | None = None,
    ) -> None:
        start_epoch = int(self.checkpoint.epoch)
        for epoch in range(start_epoch, epochs):
            total_loss = 0.0
            batch_count = 0
            for img_tensor, target in dataset:
                total_loss += float(self.train_step(img_tensor, target, start_token_id))
                batch_count += 1
            avg_epoch_loss = total_loss / batch_count if batch_count else 0.0

            self.checkpoint.epoch.assign(epoch + 1)
            logger.info("Epoch %d avg loss: %.4f", epoch + 1, avg_epoch_loss)

            if (epoch + 1) % checkpoint_interval == 0:
                save_path = self.checkpoint_manager.save()
                logger.info("Saved checkpoint at %s", save_path)

            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, avg_epoch_loss)
