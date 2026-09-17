from unittest.mock import patch

import pytest

from scripts.fetch_data import fetch_flickr8k_data


def test_exits_when_no_credentials(tmp_path):
    fake_creds = tmp_path / "missing" / "kaggle.json"
    with patch("scripts.fetch_data.KAGGLE_CREDENTIALS", fake_creds), pytest.raises(SystemExit) as exc_info:
        fetch_flickr8k_data()
    assert exc_info.value.code == 1
