from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling import feature_columns, split_xy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a trained classical model with permutation importance."
    )
    parser.add_argument("--model", default="outputs/modelos/random_forest.joblib")
    parser.add_argument("--x", default="X.csv")
    parser.add_argument("--y", default="y.csv")
    parser.add_argument("--out", default="outputs/xai")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    x_df = pd.read_csv(args.x)
    y_df = pd.read_csv(args.y)
    x_split, y_split, _ = split_xy(x_df, y_df, args.split)
    columns = feature_columns(x_df)

    result = permutation_importance(
        model,
        x_split,
        y_split,
        scoring="f1_macro",
        n_repeats=10,
        random_state=args.seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(out_dir / "permutation_importance.csv", index=False)

    top_df = importance.head(args.top)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=top_df, x="importance_mean", y="feature", color="#4C78A8")
    plt.xlabel("Queda media no F1 macro")
    plt.ylabel("Feature")
    plt.title(f"Permutation importance - {args.split}")
    plt.tight_layout()
    plt.savefig(out_dir / "permutation_importance.png", dpi=160)
    plt.close()

    print(top_df.to_string(index=False))


if __name__ == "__main__":
    main()
