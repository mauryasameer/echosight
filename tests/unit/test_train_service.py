import numpy as np
import tensorflow as tf

from src.services.train_service import CaptionTrainer


def _tiny_dataset(vocab_size: int, max_len: int = 4, n_batches: int = 2, batch_size: int = 2):
    for _ in range(n_batches):
        img = tf.random.normal((batch_size, 64, 2048))
        target = tf.constant(
            np.random.randint(1, vocab_size, size=(batch_size, max_len)), dtype=tf.int32
        )
        yield img, target


def test_trainer_starts_fresh_with_epoch_zero(tmp_path):
    trainer = CaptionTrainer(embedding_dim=32, units=64, vocab_size=50, checkpoint_dir=str(tmp_path))
    assert int(trainer.checkpoint.epoch) == 0


def test_train_step_returns_scalar_loss(tmp_path):
    trainer = CaptionTrainer(embedding_dim=32, units=64, vocab_size=50, checkpoint_dir=str(tmp_path))
    img, target = next(_tiny_dataset(vocab_size=50))
    loss = trainer.train_step(img, target, start_token_id=1)
    assert loss.shape == ()
    assert float(loss) >= 0.0


def test_fit_advances_epoch_and_checkpoints(tmp_path):
    trainer = CaptionTrainer(embedding_dim=32, units=64, vocab_size=50, checkpoint_dir=str(tmp_path))
    dataset = list(_tiny_dataset(vocab_size=50, n_batches=2))
    trainer.fit(dataset, epochs=2, start_token_id=1, checkpoint_interval=1)
    assert int(trainer.checkpoint.epoch) == 2
    assert trainer.checkpoint_manager.latest_checkpoint is not None


def test_fit_resumes_from_existing_checkpoint(tmp_path):
    trainer1 = CaptionTrainer(embedding_dim=32, units=64, vocab_size=50, checkpoint_dir=str(tmp_path))
    dataset = list(_tiny_dataset(vocab_size=50, n_batches=2))
    trainer1.fit(dataset, epochs=1, start_token_id=1, checkpoint_interval=1)

    trainer2 = CaptionTrainer(embedding_dim=32, units=64, vocab_size=50, checkpoint_dir=str(tmp_path))
    assert int(trainer2.checkpoint.epoch) == 1

    trainer2.fit(dataset, epochs=2, start_token_id=1, checkpoint_interval=1)
    assert int(trainer2.checkpoint.epoch) == 2


def test_on_epoch_end_callback_invoked(tmp_path):
    trainer = CaptionTrainer(embedding_dim=32, units=64, vocab_size=50, checkpoint_dir=str(tmp_path))
    dataset = list(_tiny_dataset(vocab_size=50, n_batches=2))
    seen: list[tuple[int, float]] = []
    trainer.fit(
        dataset, epochs=1, start_token_id=1, checkpoint_interval=1,
        on_epoch_end=lambda epoch, loss: seen.append((epoch, loss)),
    )
    assert seen == [(1, seen[0][1])]
