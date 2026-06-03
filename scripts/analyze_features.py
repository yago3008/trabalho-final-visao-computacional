from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import run_feature_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate feature analysis plots.")
    parser.add_argument("--x", default="X.csv")
    parser.add_argument("--y", default="y.csv")
    parser.add_argument("--out", default="outputs/figuras/features")
    parser.add_argument("--k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = run_feature_analysis(Path(args.x), Path(args.y), Path(args.out), k=args.k)
    print("Top selected features:")
    print(scores.head(args.k).to_string(index=False))


if __name__ == "__main__":
    main()
