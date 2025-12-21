# CA4 — Sequence Modeling (ATIS SLU)

This folder contains code to reproduce experiments for ATIS slot filling and intent detection.

Structure
- data/: download and preprocessing scripts
- src/: dataset, training, evaluation utilities
- models/: model implementations (RNN, BiRNN, BiLSTM joint, Encoder–Decoder joint)
- notebooks/: example notebooks for running experiments
- tests/: smoke tests

High-level usage
1. Install requirements: `pip install -r requirements.txt`
2. Download dataset: `python -m data.download --dataset siddhadev/atis-dataset-clean`
3. Preprocess: `python -m data.preprocess --input data/raw --out data/processed`
4. Train baseline: `python -m src.train --config configs/default.yaml --model birnn`

If you want custom tokenizer (SentencePiece / WordPiece), see `data/preprocess.py`.

