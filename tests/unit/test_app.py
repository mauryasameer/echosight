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
