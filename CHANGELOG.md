# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
Trained for real against the full Flickr8k dataset (8,091 images, ~6,500 in the
training split) on CPU, 8 epochs, batch size 32. Per-epoch average loss decreased
every epoch with no sign of divergence: 1.1555 → 0.9348 → 0.8520 → 0.7923 →
0.7427 → 0.6997 → 0.6636 → 0.6330.

End-to-end verification against two real held-out images (not used for the
BLEU/ROUGE numbers below, which are from one of them):
- `beam_caption` produced coherent, non-repeating, non-random captions on both —
  e.g. "two people are walking on a hill" and "a little girl is holding a camera"
  (the second closely matches its real reference captions; the model has learned
  a genuine, non-collapsed visual-to-text mapping, not one generic template).
- LLM enrichment (Ollama, `llama3.2`) produced a real, fluent multi-sentence
  description; real gTTS audio was generated and embedded.
- Real BLEU-4/ROUGE-L on one held-out image against its actual reference caption:
  BLEU-4 = 0.0162, ROUGE-L = 0.1053 — reported honestly, not tuned toward; this
  particular image's caption was coherent but not accurate to its content, which
  the low scores correctly reflect. No claim of a specific target score was made
  or is intended for this modest, CPU-only, 8-epoch training run.
