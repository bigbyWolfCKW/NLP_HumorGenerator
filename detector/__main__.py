import argparse
import sys


def parser():
    p = argparse.ArgumentParser(description="Detector scoring, training and QA evaluation")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List configured detectors")
    commands.add_parser("rank", add_help=False, help="Forward existing main.py rank options")
    train = commands.add_parser("train", help="Train a standard HF binary detector on QA train/validation")
    train.add_argument("--detector", default="roberta")
    train.add_argument("--train-csv")
    train.add_argument("--validation-csv")
    train.add_argument("--output", required=True, help="New checkpoint folder, never an existing model")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    train.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate", help="Evaluate a configured detector on a labelled QA CSV")
    evaluate.add_argument("--detector", default="roberta")
    evaluate.add_argument("--csv", help="Default: data/source/processed/unbiased/test.csv")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    evaluate.add_argument("--seed", type=int)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "rank":
        from main import main as pipeline
        return pipeline(argv)
    args = parser().parse_args(argv)
    if args.command == "list":
        from src.config import PROJECT
        from src.data import read_json
        for name, config in read_json(PROJECT / "configs/detectors.json").items():
            print(f"{name:24} {config['backend']:14} {config['checkpoint']}")
    elif args.command == "train":
        from .training import train_detector
        train_detector(args)
    elif args.command == "evaluate":
        if args.batch_size < 1:
            raise ValueError("Batch size must be positive")
        from .evaluation import evaluate_detector
        evaluate_detector(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
