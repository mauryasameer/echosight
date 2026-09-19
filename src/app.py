from __future__ import annotations

import argparse
import logging
import sys

from meerax.llm.claude import ClaudeProvider
from meerax.llm.ollama import OllamaProvider
from meerax.llm.openai_provider import OpenAIProvider

from src.services.caption_service import beam_caption, greedy_caption
from src.services.data_service import (
    build_annotations,
    build_tokenizer,
    load_captions,
    split_train_test,
    texts_to_padded_sequences,
)
from src.services.eval_service import evaluate_captions
from src.services.feature_service import build_feature_extractor, extract_and_cache_features
from src.services.narrative_service import enrich_caption
from src.services.report_service import build_report
from src.services.train_service import CaptionTrainer
from src.services.tts_service import text_to_speech_b64

logger = logging.getLogger(__name__)

LLM_PROVIDERS: dict[str, type] = {
    "ollama": OllamaProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
}


def _run_train(args: argparse.Namespace) -> int:
    try:
        df = load_captions(args.captions_csv, images_dir=args.images_dir)
    except FileNotFoundError as exc:
        print(f"error: could not read data file: {exc}", file=sys.stderr)
        return 1

    annotations = build_annotations(df)
    tokenizer = build_tokenizer(annotations, top_k=args.top_k)
    cap_vector = texts_to_padded_sequences(tokenizer, annotations)
    split = split_train_test(df, cap_vector, test_size=0.2)

    import pickle
    from pathlib import Path

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.checkpoint_dir}/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    extractor = build_feature_extractor()
    feature_cache = extract_and_cache_features(
        list(set(split.img_train)), extractor, cache_dir=args.feature_cache_dir
    )

    import tensorflow as tf

    trainer = CaptionTrainer(
        embedding_dim=256, units=512, vocab_size=args.top_k + 1, checkpoint_dir=args.checkpoint_dir
    )

    class _BatchedDataset:
        """Re-iterable wrapper: `trainer.fit()` iterates `dataset` once per epoch, so a
        plain generator (single-use, exhausted after the first epoch) silently yields zero
        batches on every subsequent epoch. `__iter__` returning a fresh generator each call
        makes this a genuine `Iterable`, matching `CaptionTrainer.fit()`'s own type hint."""

        def __iter__(self):
            batch_imgs, batch_caps = [], []
            for path, cap in zip(split.img_train, split.cap_train, strict=True):
                if path not in feature_cache:
                    continue
                batch_imgs.append(feature_cache[path])
                batch_caps.append(cap)
                if len(batch_imgs) == args.batch_size:
                    yield tf.convert_to_tensor(batch_imgs), tf.convert_to_tensor(batch_caps)
                    batch_imgs, batch_caps = [], []
            if batch_imgs:
                yield tf.convert_to_tensor(batch_imgs), tf.convert_to_tensor(batch_caps)

    epoch_losses: list[tuple[int, float]] = []
    trainer.fit(
        _BatchedDataset(),
        epochs=args.epochs,
        start_token_id=tokenizer.word_index["<start>"],
        checkpoint_interval=args.checkpoint_interval,
        on_epoch_end=lambda epoch, loss: epoch_losses.append((epoch, loss)),
    )

    print(f"training complete: {epoch_losses}")
    return 0


def _run_caption(args: argparse.Namespace) -> int:
    import pickle

    tokenizer_path = f"{args.checkpoint_dir}/tokenizer.pkl"
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError as exc:
        print(f"error: no trained tokenizer found at {tokenizer_path}: {exc}", file=sys.stderr)
        return 1

    # CaptionTrainer's constructor restores the latest checkpoint automatically, so its
    # encoder/decoder already carry the trained weights — no need to construct a second,
    # separate encoder/decoder pair.
    trainer_shell = CaptionTrainer(
        embedding_dim=256, units=512, vocab_size=args.top_k + 1, checkpoint_dir=args.checkpoint_dir
    )
    encoder, decoder = trainer_shell.encoder, trainer_shell.decoder

    extractor = build_feature_extractor()
    caption_fn = beam_caption if args.beam else greedy_caption
    result = caption_fn(args.image, encoder, decoder, extractor, tokenizer, max_length=args.max_length)

    raw_caption = " ".join(t for t in result.tokens if t not in ("<start>", "<end>", "<pad>"))

    llm = LLM_PROVIDERS[args.llm_provider]()
    enriched = enrich_caption(raw_caption, llm)
    audio_b64 = text_to_speech_b64(enriched)

    eval_result = None
    if args.eval and args.reference_caption:
        eval_result = evaluate_captions([args.reference_caption], [raw_caption])

    report = build_report(
        title="EchoSight — Image Description",
        image_path=args.image,
        raw_caption=raw_caption,
        enriched_description=enriched,
        audio_b64=audio_b64,
        attention_fig=None,
        eval_result=eval_result,
        epochs_trained=int(trainer_shell.checkpoint.epoch),
    )
    report.save(args.output)
    print(f"report written to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="echosight")
    parser.add_argument("--mode", choices=["train", "caption"], required=True)
    parser.add_argument("--captions-csv", default="src/data/captions.txt")
    parser.add_argument("--images-dir", default="src/data/Images")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--feature-cache-dir", default="checkpoints/features")
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--image")
    parser.add_argument("--beam", action="store_true")
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--reference-caption", default=None)
    parser.add_argument("--llm-provider", choices=list(LLM_PROVIDERS.keys()), default="ollama")
    parser.add_argument("--output", default="reports/echosight_report.html")

    # A missing required argument (e.g. --mode) makes argparse call parser.error(),
    # which raises SystemExit from inside parse_args() itself -- before any of our own
    # code runs. We want that specific case to come back as a plain int exit code (per
    # test_missing_mode_errors_cleanly, which calls main([]) directly with no
    # pytest.raises), while still letting our own later parser.error() call for a
    # missing --image propagate as SystemExit (per
    # test_caption_mode_requires_image_path, which wraps main(...) in pytest.raises).
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    if args.mode == "caption" and not args.image:
        parser.error("--image is required when --mode caption")

    if args.mode == "train":
        return _run_train(args)
    return _run_caption(args)


if __name__ == "__main__":
    raise SystemExit(main())
