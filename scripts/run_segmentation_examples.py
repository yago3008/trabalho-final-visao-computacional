from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation import compare_segmentations, read_image, resize_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save segmentation comparison examples.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--out", default="outputs/figuras/segmentacao")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        manifest[manifest["split"].eq("train")]
        .groupby("label", group_keys=False)
        .head(args.per_class)
    )
    for idx, row in selected.reset_index(drop=True).iterrows():
        image = resize_image(read_image(row["image_path"]), args.size)
        comparison = compare_segmentations(image)
        target = out_dir / f"{idx:02d}_{row['label']}_{row['original_class']}.png"
        cv2.imwrite(str(target), comparison)
    print(f"Saved {len(selected)} segmentation examples to {out_dir}")


if __name__ == "__main__":
    main()
