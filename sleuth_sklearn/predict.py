import toml

from argparse import ArgumentParser
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "PATH",
        help="The path to the config file.",
        type=str
    )
    args = parser.parse_args()

    with open(args.path, "r") as f:
        config = toml.load(f)

    