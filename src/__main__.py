import argparse

from .train.main import init_train_args


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    init_train_args(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
