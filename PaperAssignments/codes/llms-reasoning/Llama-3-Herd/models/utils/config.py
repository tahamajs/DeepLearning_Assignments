# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os
from pathlib import Path

LLAMA_MODELS_CONFIG_DIR = Path(os.getenv("LLAMA_MODELS_CONFIG_DIR", os.path.expanduser("~/.llama/")))

DEFAULT_CHECKPOINT_DIR = LLAMA_MODELS_CONFIG_DIR / "checkpoints"
