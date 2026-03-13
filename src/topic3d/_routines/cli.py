import argparse

from .train import main as run_train
from .decompose import main as run_decompose


def main():

    parser = argparse.ArgumentParser(prog="topic3d")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Register subcommands
    subparsers.add_parser("train", help="Train an LDA model with data")
    subparsers.add_parser("decompose", help="Decompose data using a trained LDA model")

    args, remaining_args = parser.parse_known_args()

    if args.command == "train":
        run_train(remaining_args)
    elif args.command == "decompose":
        run_decompose(remaining_args)