from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare classical classifiers.")
    parser.add_argument("--x", default="X.csv")
    parser.add_argument("--y", default="y.csv")
    parser.add_argument("--out", default="outputs/modelos")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        x_path=Path(args.x),
        y_path=Path(args.y),
        output_dir=Path(args.out),
        random_state=args.seed,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
