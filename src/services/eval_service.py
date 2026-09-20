from __future__ import annotations

from dataclasses import dataclass

from meerax.eval.text import bleu_score, rouge_l


@dataclass
class CaptionEvalResult:
    bleu4: float
    mean_rouge_l: float


def evaluate_captions(references: list[str], hypotheses: list[str]) -> CaptionEvalResult:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must have the same length, got {len(references)} and {len(hypotheses)}"
        )
    bleu4 = bleu_score(references, hypotheses, max_n=4)
    rouge_scores = [rouge_l(ref, hyp) for ref, hyp in zip(references, hypotheses, strict=True)]
    mean_rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    return CaptionEvalResult(bleu4=bleu4, mean_rouge_l=mean_rouge_l)
