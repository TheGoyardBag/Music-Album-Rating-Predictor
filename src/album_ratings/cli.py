import argparse
import json
from .training import train
from .inference import Predictor


def main():
    parser = argparse.ArgumentParser(description="Album ratings from descriptive tags")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--output", default="artifacts")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--max-tags", type=int, default=40)
    training.add_argument("--data-kind", choices=["synthetic", "user_supplied"],
                          default="user_supplied")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--model", default="artifacts/model.joblib")
    prediction.add_argument("--tags", required=True, help='Pipe-separated tags, e.g. "ambient|warm"')
    args = parser.parse_args()
    try:
        if args.command == "train":
            result = train(args.data, args.output, args.seed, args.max_tags, args.data_kind)
        else:
            result = Predictor(args.model).predict(args.tags.split("|"))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
