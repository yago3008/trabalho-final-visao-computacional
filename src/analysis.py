from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from src.modeling import ID_COLUMNS, feature_columns


def run_feature_analysis(
    x_path: Path,
    y_path: Path,
    output_dir: Path,
    k: int = 12,
) -> pd.DataFrame:
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    cols = feature_columns(x_df)

    train_mask = x_df["split"].eq("train")
    x_train = x_df.loc[train_mask, cols].replace([float("inf"), float("-inf")], 0).fillna(0)
    y_train = y_df.loc[train_mask, "label"]

    output_dir.mkdir(parents=True, exist_ok=True)
    merged = x_df.merge(y_df[["image_path", "label"]], on="image_path", how="left")

    scores = select_k_best(x_train, y_train, cols, k=k)
    scores.to_csv(output_dir / "select_k_best.csv", index=False)

    save_boxplots(merged, scores["feature"].head(min(k, 8)).tolist(), output_dir / "boxplots.png")
    save_correlation_heatmap(x_train[scores["feature"].head(min(k, 12))], output_dir / "correlacao.png")
    save_pca_plot(x_train, y_train, output_dir / "pca.png")
    return scores


def select_k_best(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    columns: list[str],
    k: int,
) -> pd.DataFrame:
    selector = SelectKBest(score_func=f_classif, k=min(k, len(columns)))
    selector.fit(x_train, y_train)
    scores = pd.DataFrame(
        {
            "feature": columns,
            "score": selector.scores_,
            "p_value": selector.pvalues_,
        }
    )
    return scores.sort_values("score", ascending=False)


def save_boxplots(df: pd.DataFrame, columns: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df[df["split"].eq("train")][["label", *columns]].melt(
        id_vars="label",
        var_name="feature",
        value_name="value",
    )
    g = sns.catplot(
        data=plot_df,
        x="label",
        y="value",
        col="feature",
        kind="box",
        col_wrap=4,
        sharey=False,
        height=3,
    )
    g.fig.suptitle("Boxplots das features mais discriminativas", y=1.03)
    g.savefig(output_path, dpi=160)
    plt.close(g.fig)


def save_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="vlag", center=0, square=True)
    plt.title("Correlacao entre features selecionadas")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_pca_plot(x_train: pd.DataFrame, y_train: pd.Series, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaled = StandardScaler().fit_transform(x_train)
    components = PCA(n_components=2, random_state=42).fit_transform(scaled)
    plot_df = pd.DataFrame(
        {
            "pc1": components[:, 0],
            "pc2": components[:, 1],
            "label": y_train.values,
        }
    )
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="label", alpha=0.75)
    plt.title("PCA das features manuais")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
