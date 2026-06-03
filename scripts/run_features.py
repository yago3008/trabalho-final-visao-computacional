from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import build_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract manual computer vision features.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--x-out", default="X.csv")
    parser.add_argument("--y-out", default="y.csv")
    parser.add_argument("--segmentation", choices=["hsv", "otsu"], default="hsv")
    parser.add_argument("--size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_df, y_df = build_feature_table(
        manifest_path=Path(args.manifest),
        output_x=Path(args.x_out),
        output_y=Path(args.y_out),
        segmentation_method=args.segmentation,
        image_size=args.size,
    )
    print(f"Saved {len(x_df)} feature rows to {args.x_out}")
    print(f"Saved {len(y_df)} labels to {args.y_out}")


if __name__ == "__main__":
    main()
