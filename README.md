# EchoSight

![EchoSight image-captioning to spoken-audio workflow](docs/assets/echosight-hero.png)

[![Version](https://img.shields.io/badge/version-0.1.2-blue)](CHANGELOG.md)
[![CI](https://github.com/mauryasameer/echosight/actions/workflows/ci.yml/badge.svg)](https://github.com/mauryasameer/echosight/actions)
[![Python](https://img.shields.io/badge/python-3.12-3776AB)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-informational)](LICENSE)

Image captioning (InceptionV3 → Bahdanau attention → GRU decoder, trained on Flickr8k)
with GenAI description enrichment and real text-to-speech, built on
[meerax](https://github.com/mauryasameer/the-forge). Completes the assistive use case the
source notebook's name implied but never built: caption → LLM-expanded description →
spoken audio, packaged as a self-contained HTML report.

## Setup

pip install -r requirements.txt

## Data

python scripts/fetch_data.py   # requires a Kaggle API token at ~/.kaggle/kaggle.json

## Usage

python -m src.app --mode train --epochs 10
python -m src.app --mode caption --image path/to/image.jpg --beam
python -m src.app --mode caption --image path/to/image.jpg --eval --reference-caption "a dog running in the grass"

## 🐳 Docker

docker compose build
docker compose run echosight python -m src.app --mode caption --image src/data/Images/example.jpg

## ⚖️ Governance

See [GOVERNANCE.md](./GOVERNANCE.md) for intended use, the explainability boundary,
fairness limitations, LLM controls, audit trail, and regulatory framing.

## License

Distributed under the MIT License. See `LICENSE` for more information.
