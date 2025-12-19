# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import argparse

from .describe import Describe
from .download import Download
from .list import List
from .prompt_format import PromptFormat
from .remove import Remove
from .utils import print_subcommand_description
from .verify_download import VerifyDownload


class LlamaModelsCLIParser:
    """Defines CLI parser for Llama Models CLI"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="llama-model",
            description="Llama Model Management CLI",
            add_help=True,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Default command is to print help
        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        subparsers = self.parser.add_subparsers(title="subcommands")

        # Add sub-commands
        List.create(subparsers)
        Describe.create(subparsers)
        Download.create(subparsers)
        PromptFormat.create(subparsers)
        Remove.create(subparsers)
        VerifyDownload.create(subparsers)

        print_subcommand_description(self.parser, subparsers)

    def parse_args(self) -> argparse.Namespace:
        args = self.parser.parse_args()
        if not isinstance(args, argparse.Namespace):
            raise TypeError(f"Expected argparse.Namespace, got {type(args)}")
        return args

    def run(self, args: argparse.Namespace) -> None:
        args.func(args)


def main():
    parser = LlamaModelsCLIParser()
    args = parser.parse_args()
    parser.run(args)


if __name__ == "__main__":
    main()
