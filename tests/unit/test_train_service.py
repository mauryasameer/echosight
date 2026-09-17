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


def test_resume_restores_encoder_and_attention_weights_and_keeps_training(tmp_path):
    """Regression test for the deferred-restore bug: encoder/attention variables are
    only built lazily on first call(), so restoring a checkpoint before ever calling
    encoder/decoder used to silently leave encoder.fc and attention.W1/W2 at their
    random-init values (no error raised) and then those variables never received
    gradients again. _warm_build() must run before _restore_latest() so the restore
    actually reconnects every variable."""
    trainer1 = CaptionTrainer(embedding_dim=32, units=64, vocab_size=50, checkpoint_dir=str(tmp_path))
    dataset = list(_tiny_dataset(vocab_size=50, n_batches=2))
    trainer1.fit(dataset, epochs=1, start_token_id=1, checkpoint_interval=1)

    encoder_fc_kernel_ref = trainer1.encoder.fc.kernel.numpy().copy()
    attention_w1_kernel_ref = trainer1.decoder.attention.W1.kernel.numpy().copy()

    trainer2 = CaptionTrainer(embedding_dim=32, units=64, vocab_size=50, checkpoint_dir=str(tmp_path))

    encoder_fc_match = np.allclose(trainer2.encoder.fc.kernel.numpy(), encoder_fc_kernel_ref)
    attention_w1_match = np.allclose(trainer2.decoder.attention.W1.kernel.numpy(), attention_w1_kernel_ref)
    assert encoder_fc_match, "encoder.fc.kernel was not restored (came back as random-init)"
    assert attention_w1_match, "attention.W1.kernel was not restored (came back as random-init)"

    img, target = next(_tiny_dataset(vocab_size=50))
    hidden = trainer2.decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([1] * target.shape[0], 1)
    with tf.GradientTape() as tape:
        features = trainer2.encoder(img)
        predictions, hidden, _ = trainer2.decoder(dec_input, features, hidden)
        loss = tf.reduce_mean(predictions)
    gradients = tape.gradient(loss, trainer2.encoder.trainable_variables)
    assert all(g is not None for g in gradients), "encoder is silently frozen after resume"
    assert any(float(tf.norm(g)) > 0.0 for g in gradients), "encoder gradients are all zero after resume"
