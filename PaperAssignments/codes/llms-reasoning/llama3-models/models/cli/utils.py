# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.


def print_subcommand_description(parser, subparsers):
    """Print descriptions of subcommands."""
    description_text = ""
    for name, subcommand in subparsers.choices.items():
        description = subcommand.description
        description_text += f"  {name:<21} {description}\n"
    parser.epilog = description_text
