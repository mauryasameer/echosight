import pytest

from src.services.eval_service import evaluate_captions


def test_perfect_match_scores_high():
    references = ["a dog running in the grass", "a cat sleeping on a couch"]
    hypotheses = ["a dog running in the grass", "a cat sleeping on a couch"]
    result = evaluate_captions(references, hypotheses)
    assert result.bleu4 == pytest.approx(1.0, abs=0.05)
    assert result.mean_rouge_l == pytest.approx(1.0, abs=0.01)


def test_completely_different_scores_low():
    references = ["a dog running in the grass"]
    hypotheses = ["a submarine underwater near coral"]
    result = evaluate_captions(references, hypotheses)
    assert result.bleu4 < 0.3
    assert result.mean_rouge_l < 0.3


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        evaluate_captions(["one", "two"], ["only one"])
