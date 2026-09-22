# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] - 2026-09-22

The v0.2.0 backlog logged in `[0.1.1]`'s Known Limitations, all closed with real code
fixes and a real retrain (not just documentation this time).

### Fixed
- `split_train_test` now groups by image, not caption row -- the actual code fix for
  the data leakage `[0.1.1]` could only correct in the docs. Verified against the real
  dataset: 6,472 train images / 1,619 test images, **zero overlap**.
- `CNN_Encoder`'s `Dropout(0.5)` is now actually applied during training (was
  constructed but never called) -- real regularization in this run, unlike `v0.1.0`/
  `v0.1.1`.
- `--mode caption` now refuses to run (exit 1, clear error) against a checkpoint
  directory with no actual checkpoint, instead of silently captioning on random-init
  weights and still exiting 0.

### Added
- Real attention maps are now rendered in the HTML report for greedy captions (one
  subplot per generated word, the model's real attention weights overlaid on the
  image) -- previously computed by `greedy_caption` but discarded. `beam_caption`
  still gets no attention figure; it only ever returns a placeholder, not real
  per-step weights, so rendering it would show a meaningless map as if it were real.
- Real end-to-end test coverage for both CLI paths (`_run_caption`, `_run_train`)
  against a real, small, genuinely-trained checkpoint -- closes the exact class of
  gap (untested real wiring) that let two real bugs ship in `v0.1.0`/`v0.1.1`.

### Real retrain results (genuinely held-out this time)
Retrained on CPU, 8 epochs, batch size 32, same Flickr8k data and feature cache as
before (no re-download/re-extraction needed -- only the split logic changed, so the
cached per-image features are still valid). Loss decreased every epoch, with dropout
now genuinely active: 1.2050 → 0.9876 → 0.9141 → 0.8645 → 0.8255 → 0.7925 → 0.7619 →
0.7345.

Evaluated against a random sample of 10 real, **genuinely held-out** images (verified
zero overlap with the training set, unlike `v0.1.0`'s numbers) using `greedy_caption`:
BLEU-4 = 0.0195, mean ROUGE-L = 0.2971. These are honestly lower than `v0.1.0`'s
(train-contaminated, since retracted) numbers -- an 8-epoch, CPU-only, modest-capacity
model genuinely does not generalize well yet; it frequently defaults to common
patterns ("a dog is running through the grass") rather than precise, diverse
descriptions, and on one sampled image `greedy_caption` fell into a real repetition
loop ("...wearing a red jacket is wearing a red jacket...") -- a known failure mode of
unconstrained greedy decoding on an undertrained model, not a code defect (its hidden
state threading is correct and verified; this is a decoding-strategy limitation, not
the `v0.1.1` hidden-state bug recurring). These numbers are reported honestly, exactly
as produced, with no cherry-picking and no claim of a target score.

### Known limitations (tracked for v0.3.0, not fixed in this release)
- Greedy decoding has no repetition guard and can loop on some images (see above);
  beam search's own lack of length normalization (noted in `v0.1.1`) may also
  contribute. Both are decoding-strategy issues, addressable without retraining.
- More epochs, a larger model, or a larger training subset would likely improve the
  honestly-low BLEU-4/ROUGE-L above -- not attempted here to keep this retrain's
  wall-clock/disk footprint modest, consistent with the project's CPU-only, small-
  scale scope.

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
