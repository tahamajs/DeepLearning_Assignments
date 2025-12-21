#!/usr/bin/env bash
set -e

# Example commands to run smoke tests for each CA
python scripts/download_datasets.py --urbansound --flickr || true
python -c "from q1_image_captioning import tokenizer; print('tokenizer ok')"
python -c "from q1_clip import models; print('clip models ok')"
python -c "from q2_urban_sound import data; print('urban sound ok')"
python -c "from q3_lora import train; print('lora ok')"
python -c "from q4_adversarial import attacks; print('attacks ok')"
echo 'Smoke run completed'