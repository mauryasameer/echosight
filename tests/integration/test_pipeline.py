import ast
import pickle
from unittest.mock import MagicMock

import numpy as np
import tensorflow as tf
from meerax.llm.base import LLMResponse
from PIL import Image

import src.app as app_module
from src.services.eval_service import evaluate_captions
from src.services.narrative_service import enrich_caption
from src.services.report_service import build_report
from src.services.train_service import CaptionTrainer
from src.services.tts_service import text_to_speech_b64


class _StubLLM:
    def generate(self, prompt, system=None, **kwargs):
        return LLMResponse(
            content="A small brown dog is running through a green field.",
            model="stub", input_tokens=1, output_tokens=1,
        )

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def test_full_pipeline_produces_nonempty_report(tmp_path, monkeypatch):
    # Exercise caption -> enrich -> TTS -> eval -> report end to end with a synthetic
    # caption (not a real trained checkpoint or real image) and a stubbed LLM.
    raw_caption = "dog running grass"
    enriched = enrich_caption(raw_caption, _StubLLM())
    assert enriched == "A small brown dog is running through a green field."

    audio_b64 = text_to_speech_b64(enriched)
    assert len(audio_b64) > 50

    eval_result = evaluate_captions(
        references=["a dog runs through the grass"], hypotheses=[raw_caption]
    )

    report = build_report(
        title="EchoSight Test Report",
        image_path="test_image.jpg",
        raw_caption=raw_caption,
        enriched_description=enriched,
        audio_b64=audio_b64,
        attention_fig=None,
        eval_result=eval_result,
        epochs_trained=1,
    )

    output_path = tmp_path / "report.html"
    report.save(str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


class _TinyTokenizer:
    def __init__(self):
        self.word_index = {
            "<pad>": 0, "<start>": 1, "<end>": 2, "a": 3, "dog": 4,
            "runs": 5, "cat": 6, "sleeps": 7, "on": 8, "grass": 9,
        }
        self.index_word = {v: k for k, v in self.word_index.items()}


def test_run_caption_cli_end_to_end_against_a_real_checkpoint(tmp_path, monkeypatch):
    """Regression test for the exact class of gap that let both real bugs from
    EchoSight's real training run (Task 13) ship past every per-task review: this
    file's other test never touched caption_service, feature_service, or
    CaptionTrainer at all -- it built a report from a hardcoded string. Everything
    downstream of a real checkpoint restore was untested.

    This runs the actual CLI entry point (`src.app.main(["--mode", "caption", ...])`)
    against a real, small, genuinely-trained checkpoint -- built with the SAME
    embedding_dim=256/units=512 `_run_caption` hardcodes, since those aren't
    configurable via CLI and a mismatched checkpoint would fail to restore. Only the
    feature extractor and image decoding are mocked (avoids a real InceptionV3 weight
    load, matching the pattern already used in test_feature_service.py and
    test_caption_service.py) and the LLM provider is stubbed (avoids a real network
    dependency, matching this file's existing test) -- gTTS still makes a real network
    call, same as the existing test."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    vocab_size = 10

    trainer = CaptionTrainer(
        embedding_dim=256, units=512, vocab_size=vocab_size, checkpoint_dir=str(checkpoint_dir)
    )
    # One real (tiny) training step so the checkpoint reflects genuinely-trained
    # weights, not just the constructor's random init.
    fake_dataset = [(
        tf.random.normal((2, 64, 2048)),
        tf.constant(np.random.randint(1, vocab_size, size=(2, 4)), dtype=tf.int32),
    )]
    trainer.fit(fake_dataset, epochs=1, start_token_id=1, checkpoint_interval=1)

    with open(checkpoint_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(_TinyTokenizer(), f)

    # A real (if tiny/synthetic) image file: build_attention_figure() opens it for
    # real (it isn't covered by the load_and_preprocess_image mock below, which only
    # bypasses feature_service's real InceptionV3 preprocessing path).
    image_path = tmp_path / "test_image.jpg"
    Image.fromarray((np.random.rand(50, 50, 3) * 255).astype(np.uint8)).save(image_path)

    fake_extractor = MagicMock(return_value=tf.random.normal((1, 8, 8, 2048)))
    monkeypatch.setattr("src.app.build_feature_extractor", lambda: fake_extractor)
    monkeypatch.setattr(
        "src.services.caption_service.load_and_preprocess_image",
        lambda path: tf.zeros((299, 299, 3)),
    )
    monkeypatch.setitem(app_module.LLM_PROVIDERS, "ollama", lambda: _StubLLM())

    output_path = tmp_path / "report.html"
    exit_code = app_module.main([
        "--mode", "caption",
        "--image", str(image_path),
        "--checkpoint-dir", str(checkpoint_dir),
        "--top-k", str(vocab_size - 1),
        "--output", str(output_path),
    ])

    assert exit_code == 0
    assert output_path.exists()
    html = output_path.read_text()
    assert "<audio controls" in html
    assert "Run Info" in html
    assert "epochs trained: 1" in html
    assert "data:image/png;base64," in html, "greedy captioning must embed a real attention figure"


def test_run_train_cli_produces_real_loss_every_epoch(tmp_path, monkeypatch, capsys):
    """Regression test for the exact class of gap that shipped the real generator-
    exhaustion bug (fixed in v0.1.1, PR #12): _run_train's `_BatchedDataset` was, until
    that fix, a plain one-shot generator passed to `trainer.fit()`, which iterates the
    same dataset object once per epoch. Epoch 1 exhausted it for real; every later
    epoch silently iterated zero batches and reported loss=0.0 via `fit()`'s
    `if batch_count else 0.0` guard -- no test ever caught this because the fix
    landed without one. This runs the real `--mode train` CLI path for 3 epochs
    against a tiny real dataset and asserts every epoch has a genuine nonzero loss,
    not just epoch 1."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    captions_csv = tmp_path / "captions.txt"
    lines = ["image,caption"]
    for i in range(6):
        lines.append(f"img{i}.jpg,a tiny caption number {i}")
        lines.append(f"img{i}.jpg,another caption for image {i}")
    captions_csv.write_text("\n".join(lines) + "\n")

    fake_extractor = MagicMock(return_value=tf.random.normal((1, 8, 8, 2048)))
    monkeypatch.setattr("src.app.build_feature_extractor", lambda: fake_extractor)
    monkeypatch.setattr(
        "src.services.feature_service.load_and_preprocess_image",
        lambda path: tf.zeros((299, 299, 3)),
    )

    checkpoint_dir = tmp_path / "checkpoints"
    exit_code = app_module.main([
        "--mode", "train",
        "--captions-csv", str(captions_csv),
        "--images-dir", str(images_dir),
        "--checkpoint-dir", str(checkpoint_dir),
        "--feature-cache-dir", str(tmp_path / "features"),
        "--top-k", "9",
        "--batch-size", "4",
        "--epochs", "3",
        "--checkpoint-interval", "1",
    ])
    assert exit_code == 0

    captured = capsys.readouterr()
    line = next(ln for ln in captured.out.splitlines() if ln.startswith("training complete: "))
    epoch_losses = ast.literal_eval(line.removeprefix("training complete: "))

    assert len(epoch_losses) == 3
    for epoch, loss in epoch_losses:
        assert loss > 0.0, f"epoch {epoch} had loss {loss} -- dataset was likely exhausted after epoch 1"
