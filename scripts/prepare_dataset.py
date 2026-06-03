from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import (
    balanced_sample,
    choose_group_representatives,
    extract_entries,
    list_dataset_entries,
    split_train_val,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a balanced fresh/rotten sample from the Kaggle ZIP."
    )
    parser.add_argument(
        "--zip",
        default="data/raw/fruits_fresh_rotten.zip",
        help="Path to the Kaggle dataset ZIP.",
    )
    parser.add_argument(
        "--out",
        default="data/processed",
        help="Directory where sampled images and manifest.csv will be written.",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=500,
        help="Number of original train images per binary class before validation split.",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=150,
        help="Number of test images per binary class.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of sampled train images moved to validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-augmentations",
        action="store_true",
        help="Keep augmented variants instead of selecting one image per source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip)
    output_dir = Path(args.out)
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")

    entries = list_dataset_entries(zip_path)
    if not args.keep_augmentations:
        entries = choose_group_representatives(entries, seed=args.seed)

    sampled_train = []
    sampled_test = []
    for label in ("fresh", "rotten"):
        sampled_train.extend(
            balanced_sample(
                entries,
                split="train",
                label=label,
                limit=args.train_per_class,
                seed=args.seed,
            )
        )
        sampled_test.extend(
            balanced_sample(
                entries,
                split="test",
                label=label,
                limit=args.test_per_class,
                seed=args.seed + 1,
            )
        )

    train_final, val_final = split_train_val(
        sampled_train,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    rows = []
    rows.extend(extract_entries(zip_path, output_dir, "train", train_final))
    rows.extend(extract_entries(zip_path, output_dir, "val", val_final))
    rows.extend(extract_entries(zip_path, output_dir, "test", sampled_test))
    write_manifest(rows, output_dir / "manifest.csv")

    print("Dataset sample prepared")
    print(f"manifest: {output_dir / 'manifest.csv'}")
    print(f"train: {len(train_final)} images")
    print(f"val:   {len(val_final)} images")
    print(f"test:  {len(sampled_test)} images")


if __name__ == "__main__":
    main()
