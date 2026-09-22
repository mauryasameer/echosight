# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.2] - 2026-09-22

### Changed
- Added a repository hero that illustrates EchoSight's image-to-caption-to-audio assistive workflow.

## [0.1.1] - 2026-09-20

### Fixed
- `_run_caption` constructed the LLM provider (`LLM_PROVIDERS[args.llm_provider]()`)
  outside `enrich_caption`'s failure-isolation boundary, so a provider-construction
  failure (e.g. a missing API key env var for `--llm-provider claude`/`openai`)
  crashed the whole CLI with a raw traceback instead of falling back to the raw
  caption like every other enrichment failure. Provider construction now happens
  inside the same try/except as the enrichment call.

### Corrected (documentation)
- `[0.1.0]`'s "real training results" section originally overstated the training
  split and held-out verification — see the correction inline in that entry below.
  The train/test split leaks nearly all images between splits (row-level split on
  a 5-captions-per-image dataset); a proper fix requires an image-grouped split
  and a retrain, tracked for `v0.2.0`.
- `GOVERNANCE.md`'s Explainability section claimed attention maps are "directly
  inspectable" in every report. They are computed by `greedy_caption` but never
  rendered (`attention_fig` is hardcoded to `None` in `src/app.py`, and
  `beam_caption` — available via `--beam` — only ever returns a zero-filled
  placeholder, not real attention weights). Corrected to accurately describe the
  current state; wiring a real attention visualization through is tracked for
  `v0.2.0`.

### Known limitations (tracked for v0.2.0, not fixed in this release)
- Train/test split is not grouped by image (see above) — affects reported
  evaluation metrics, not the model's ability to produce coherent captions.
- `CNN_Encoder`'s `Dropout(0.5)` layer is constructed but never applied in
  `call()` — the real training run above had no regularization.
- `--mode caption` against a checkpoint directory with no actual checkpoint
  silently starts from random-init weights, produces a garbage caption, and
  still writes and exits 0 — should refuse or warn loudly instead.
- `beam_caption` has no length normalization (short-caption bias) and no
  end-to-end test coverage against a real/synthetic trained checkpoint — the
  same class of gap that let both `v0.1.0` real-training bugs through per-task
  review undetected.

## [0.1.0] - 2026-09-20

### Added
- Image captioning pipeline (InceptionV3 frozen CNN encoder → Bahdanau attention →
  GRU decoder with teacher forcing), trained on Flickr8k, with both greedy and
  beam-search inference (`beam_caption`/`greedy_caption`).
- Resumable training (`CaptionTrainer`, `tf.train.CheckpointManager`, epoch tracked
  as a persisted `tf.Variable`) — safe to stop and restart across sessions.
- GenAI caption enrichment (`narrative_service.py`) via `meerax.llm`
  (Ollama/Claude/OpenAI), `temperature=0` for reproducibility, falling back to the
  raw caption itself (not a placeholder) on LLM failure.
- Real text-to-speech (`tts_service.py`, gTTS) producing in-memory MP3 audio,
  embedded directly in the HTML report via a base64 `<audio>` tag.
- Evaluation (`eval_service.py`) — real BLEU-4/ROUGE-L against reference captions
  via `meerax.eval.text`.
- Self-contained HTML report (`report_service.py`) via `meerax.report.builder`,
  with a permanent governance banner, escaped user/model-controlled content, and
  an optional Evaluation section.
- `scripts/fetch_data.py` — Kaggle Flickr8k dataset fetch, requires a Kaggle API
  token, fails loudly without one.
- CLI entry point (`src/app.py`, `--mode train` / `--mode caption`).
- Docker packaging (`Dockerfile`, `docker-compose.yml`), `GOVERNANCE.md`.

### Fixed
- `_run_train` passed a one-shot Python generator to `CaptionTrainer.fit()`, which
  iterates the same dataset object once per epoch — silently zeroed loss from
  epoch 2 onward. Found during the real training run below; fixed by making the
  batch source a genuine re-iterable.
- `beam_caption` discarded the decoder's hidden state on every step, reusing the
  same static initial state for the whole stateful GRU decode — produced infinite
  repeating-token captions on a real trained model (never seen in per-task review,
  since its own tests only ever mocked a decoder that ignores hidden state).
  `greedy_caption` was unaffected (it threads hidden state correctly). Fixed by
  giving each beam candidate its own threaded hidden state.

### Real training results
Trained for real against the full Flickr8k dataset (8,091 images, 32,364 caption
rows) on CPU, 8 epochs, batch size 32. Per-epoch average loss decreased every
epoch with no sign of divergence: 1.1555 → 0.9348 → 0.8520 → 0.7923 → 0.7427 →
0.6997 → 0.6636 → 0.6330.

**Correction (found by the final whole-branch review, after this entry was first
published — see `[0.1.1]` below): the "two real held-out images" and "~6,500 in
the training split" claims originally written here were wrong.** `split_train_test`
splits caption *rows*, not images; since each image has 5 caption rows, this
leaks nearly every image into both splits. Only 1 of the dataset's 8,091 images
is genuinely unseen by training — the model was effectively trained on 8,090 of
them (99.99%). The BLEU-4/ROUGE-L numbers originally reported here (BLEU-4 = 0.0162,
ROUGE-L = 0.1053, since removed) were therefore
train-set-contaminated, not held-out generalization scores. `beam_caption` and
`greedy_caption` do produce coherent, non-repeating, non-collapsed captions (that
part holds up — verified independently on multiple images including the one
genuinely held-out image), but the accuracy/eval numbers should not be read as
evidence of generalization. A proper image-grouped split, retrain, and honest
re-report of held-out metrics is tracked for `v0.2.0`.

LLM enrichment (Ollama, `llama3.2`) produced a real, fluent multi-sentence
description on real captions; real gTTS audio was generated and embedded; the
HTML report rendered correctly with escaped content and a governance banner.
