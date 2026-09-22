import pickle

import pytest

from src.app import main


def test_missing_mode_errors_cleanly(capsys):
    exit_code = main([])
    assert exit_code != 0


def test_caption_mode_requires_image_path(capsys):
    with pytest.raises(SystemExit):
        main(["--mode", "caption"])


def test_train_mode_requires_data_present(tmp_path, capsys):
    exit_code = main([
        "--mode", "train",
        "--captions-csv", str(tmp_path / "missing.txt"),
        "--images-dir", str(tmp_path / "missing"),
        "--epochs", "1",
    ])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err


def test_caption_mode_refuses_without_a_real_checkpoint(tmp_path, capsys):
    """Regression test: --mode caption used to silently run on random-init weights
    when the checkpoint directory had a tokenizer.pkl but no actual checkpoint files
    (e.g. a fresh --checkpoint-dir that was never trained into), producing a garbage
    caption, real synthesized audio for it, and a written report -- all while exiting
    0 as if it had succeeded. It must now refuse."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    with open(checkpoint_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump({"word_index": {"<pad>": 0}, "index_word": {0: "<pad>"}}, f)

    output_path = tmp_path / "report.html"
    exit_code = main([
        "--mode", "caption",
        "--image", "/fake/image.jpg",
        "--checkpoint-dir", str(checkpoint_dir),
        "--output", str(output_path),
    ])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err
    assert "checkpoint" in captured.err
    assert not output_path.exists(), "must not write a report from an untrained model"
