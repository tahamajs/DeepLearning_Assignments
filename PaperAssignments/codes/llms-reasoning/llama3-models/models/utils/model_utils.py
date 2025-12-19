# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from pathlib import Path

from .config import DEFAULT_CHECKPOINT_DIR


def model_local_dir(descriptor: str) -> str:
    """Get the local directory path for a model given its descriptor."""
    return str(Path(DEFAULT_CHECKPOINT_DIR) / (descriptor.replace(":", "-")))
