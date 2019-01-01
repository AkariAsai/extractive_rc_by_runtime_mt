from typing import Dict
import argparse
import logging
import sys

from allennlp.common.util import import_submodules
from allennlp.commands.fine_tune import FineTune
from allennlp.commands.make_vocab import MakeVocab
from allennlp.commands.subcommand import Subcommand

from evaluate import Evaluate
from evaluate_mlqa import Evaluate_MLQA
from train import Train


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main():
    prog = "python -m allennlp.run"
    subcommand_overrides = {}
    # pylint: disable=dangerous-default-value
    parser = argparse.ArgumentParser(
        description="Run AllenNLP", usage='%(prog)s', prog=prog)
    print(parser)

    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
        # Default commands
        "train": Train(),
        "evaluate": Evaluate(),
        "evaluate_mlqa": Evaluate_MLQA(),
        "make-vocab": MakeVocab(),
        "fine-tune": FineTune(),
        # Superseded by overrides
        **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--include-package',
                               type=str,
                               action='append',
                               default=[],
                               help='additional packages to include')

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in args.include_package:
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
