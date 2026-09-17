import numpy as np
import pytest

from src.services.data_service import build_annotations, build_tokenizer, load_captions, split_train_test


@pytest.fixture
def captions_csv(tmp_path):
    path = tmp_path / "captions.txt"
    path.write_text(
        "image,caption\n"
        "img1.jpg,a dog running in the grass\n"
        "img1.jpg,a brown dog runs outside\n"
        "img2.jpg,a cat sleeping on a couch\n"
    )
    return str(path)


def test_load_captions_returns_id_path_captions_columns(captions_csv):
    df = load_captions(captions_csv, images_dir="/fake/images")
    assert list(df.columns) == ["ID", "Path", "Captions"]
    assert len(df) == 3
    assert df.iloc[0]["ID"] == "img1"
    assert df.iloc[0]["Path"] == "/fake/images/img1.jpg"


def test_build_annotations_wraps_start_end(captions_csv):
    df = load_captions(captions_csv, images_dir="/fake/images")
    annotations = build_annotations(df)
    assert annotations[0].startswith("<start> ")
    assert annotations[0].endswith(" <end>")
    assert len(annotations) == 3


def test_build_tokenizer_maps_pad_to_zero(captions_csv):
    df = load_captions(captions_csv, images_dir="/fake/images")
    annotations = build_annotations(df)
    tokenizer = build_tokenizer(annotations, top_k=50)
    assert tokenizer.word_index["<pad>"] == 0
    assert tokenizer.index_word[0] == "<pad>"


def test_split_train_test_shapes(captions_csv):
    df = load_captions(captions_csv, images_dir="/fake/images")
    cap_vector = np.zeros((3, 10), dtype=np.int32)
    result = split_train_test(df, cap_vector, test_size=0.34)
    assert len(result.img_train) + len(result.img_test) == 3
    assert result.cap_train.shape[0] == len(result.img_train)
    assert result.cap_test.shape[0] == len(result.img_test)
