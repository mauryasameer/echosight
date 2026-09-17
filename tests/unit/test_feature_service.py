from pathlib import Path
from unittest.mock import MagicMock

import pytest
import tensorflow as tf

from src.services.feature_service import extract_and_cache_features


def test_extract_and_cache_features_writes_cache_files(tmp_path):
    fake_extractor = MagicMock()
    fake_extractor.return_value = tf.random.normal((1, 8, 8, 2048))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.services.feature_service.load_and_preprocess_image",
            lambda path: tf.zeros((299, 299, 3)),
        )
        result = extract_and_cache_features(
            ["/fake/img1.jpg"], fake_extractor, cache_dir=str(tmp_path)
        )

    assert "/fake/img1.jpg" in result
    assert result["/fake/img1.jpg"].shape == (64, 2048)
    cache_files = list(Path(tmp_path).glob("*.pkl"))
    assert len(cache_files) == 1


def test_extract_and_cache_features_reuses_cache_on_second_call(tmp_path):
    fake_extractor = MagicMock()
    fake_extractor.return_value = tf.random.normal((1, 8, 8, 2048))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.services.feature_service.load_and_preprocess_image",
            lambda path: tf.zeros((299, 299, 3)),
        )
        extract_and_cache_features(["/fake/img1.jpg"], fake_extractor, cache_dir=str(tmp_path))
        call_count_after_first = fake_extractor.call_count

        extract_and_cache_features(["/fake/img1.jpg"], fake_extractor, cache_dir=str(tmp_path))
        call_count_after_second = fake_extractor.call_count

    # second call should hit the cache, not re-invoke the extractor
    assert call_count_after_second == call_count_after_first
