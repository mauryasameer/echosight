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


@pytest.fixture
def multi_caption_csv(tmp_path):
    """5 captions per image (Flickr8k's real shape), 10 images -- large enough that a
    row-level (rather than image-grouped) split would, with overwhelming probability,
    put at least one image's captions on both sides."""
    path = tmp_path / "captions.txt"
    lines = ["image,caption"]
    for i in range(10):
        for j in range(5):
            lines.append(f"img{i}.jpg,caption {j} for image {i}")
    path.write_text("\n".join(lines) + "\n")
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


def test_split_train_test_does_not_leak_images_across_splits(multi_caption_csv):
    """Regression test for a real bug found by EchoSight's v0.1.0 final review: the
    split used to operate on caption ROWS, not images. Since Flickr8k has 5 captions
    per image, that let the same image's captions land on both sides -- 8,090 of
    8,091 real images leaked into both train and test, and the reported BLEU/ROUGE
    scores were train-set-contaminated (v0.1.0's published CHANGELOG/Release claims
    about "held-out" verification were factually wrong; corrected in v0.1.1). This
    fixture's 10 images x 5 captions mirrors that shape at a size where a row-level
    split would leak with near certainty."""
    df = load_captions(multi_caption_csv, images_dir="/fake/images")
    cap_vector = np.zeros((len(df), 10), dtype=np.int32)
    result = split_train_test(df, cap_vector, test_size=0.3)

    def image_id(path: str) -> str:
        return path.rsplit("/", 1)[-1].removesuffix(".jpg")

    train_ids = {image_id(p) for p in result.img_train}
    test_ids = {image_id(p) for p in result.img_test}
    assert train_ids.isdisjoint(test_ids), (
        f"image(s) leaked into both splits: {train_ids & test_ids}"
    )
    # every image's captions stay together: an image in train has ALL 5 of its rows there
    for image in train_ids:
        assert result.img_train.count(f"/fake/images/{image}.jpg") == 5
    for image in test_ids:
        assert result.img_test.count(f"/fake/images/{image}.jpg") == 5
