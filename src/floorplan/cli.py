import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate floor-plan segmentation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--raw", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=512)
    p = commands.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p = commands.add_parser("infer")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", dest="image_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--size", type=int, default=512)
    p = commands.add_parser("evaluate")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", default="auto")
    p.add_argument("--samples", type=int, default=8)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from .data import prepare

        result = prepare(**args)
    elif command == "train":
        from .train import train

        result = train(**args)
    elif command == "infer":
        from .evaluate import infer

        result = infer(**args)
    else:
        from .evaluate import evaluate

        result = evaluate(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
