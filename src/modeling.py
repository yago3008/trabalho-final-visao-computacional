from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_curve,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ID_COLUMNS = {"image_path", "split", "original_class"}
LABEL_ORDER = ["fresh", "rotten"]


def load_xy(x_path: Path, y_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    return x_df, y_df


def feature_columns(x_df: pd.DataFrame) -> list[str]:
    return [col for col in x_df.columns if col not in ID_COLUMNS]


def split_xy(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    split: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cols = feature_columns(x_df)
    idx = x_df["split"].eq(split)
    x_split = x_df.loc[idx, cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_split = y_df.loc[idx, "label"]
    paths = x_df.loc[idx, "image_path"]
    return x_split, y_split, paths


def model_grids(random_state: int) -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__solver": ["lbfgs"],
            },
        ),
        "svm_rbf": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(class_weight="balanced", probability=True)),
                ]
            ),
            {
                "clf__C": [0.5, 1.0, 5.0],
                "clf__gamma": ["scale", 0.01, 0.001],
            },
        ),
        "random_forest": (
            Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 8, 16],
                "clf__min_samples_leaf": [1, 3],
            },
        ),
    }


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> dict[str, float | str]:
    return {
        "split": prefix,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_rotten": precision_score(
            y_true,
            y_pred,
            labels=["rotten"],
            average="macro",
            zero_division=0,
        ),
        "recall_rotten": recall_score(
            y_true,
            y_pred,
            labels=["rotten"],
            average="macro",
            zero_division=0,
        ),
        "f1_rotten": f1_score(
            y_true,
            y_pred,
            labels=["rotten"],
            average="macro",
            zero_division=0,
        ),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
    normalize: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER, normalize=normalize)
    display = ConfusionMatrixDisplay(cm, display_labels=LABEL_ORDER)
    display.plot(cmap="Blues", values_format=".2f" if normalize else "d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_roc_curve(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positive = (y_test == "rotten").astype(int).to_numpy()
    classes = list(model.classes_)
    rotten_index = classes.index("rotten")
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test)[:, rotten_index]
    else:
        scores = model.decision_function(x_test)
    fpr, tpr, _ = roc_curve(positive, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.title("Curva ROC - classe rotten")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def prediction_table(
    model_name: str,
    model: Pipeline,
    split_name: str,
    x_split: pd.DataFrame,
    y_split: pd.Series,
    paths: pd.Series,
) -> pd.DataFrame:
    pred = model.predict(x_split)
    rows = pd.DataFrame(
        {
            "model": model_name,
            "split": split_name,
            "image_path": paths.values,
            "y_true": y_split.values,
            "y_pred": pred,
        }
    )
    rows["correct"] = rows["y_true"].eq(rows["y_pred"])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_split)
        for label in LABEL_ORDER:
            if label in model.classes_:
                label_index = list(model.classes_).index(label)
                rows[f"proba_{label}"] = probabilities[:, label_index]
        probability_columns = [f"proba_{label}" for label in LABEL_ORDER if f"proba_{label}" in rows]
        if probability_columns:
            rows["confidence"] = rows[probability_columns].max(axis=1)
    return rows


def train_and_evaluate(
    x_path: Path,
    y_path: Path,
    output_dir: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    x_df, y_df = load_xy(x_path, y_path)
    x_train, y_train, _ = split_xy(x_df, y_df, "train")
    x_val, y_val, val_paths = split_xy(x_df, y_df, "val")
    x_test, y_test, test_paths = split_xy(x_df, y_df, "test")

    output_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, float | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    best_models: dict[str, Pipeline] = {}

    for model_name, (pipeline, grid) in model_grids(random_state).items():
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_models[model_name] = model
        joblib.dump(model, output_dir / f"{model_name}.joblib")

        for split_name, x_split, y_split in [
            ("validation", x_val, y_val),
            ("test", x_test, y_test),
        ]:
            pred = model.predict(x_split)
            row = calculate_metrics(y_split, pred, split_name)
            row["model"] = model_name
            row["best_params"] = str(search.best_params_)
            rows.append(row)

        for split_name, x_split, y_split, paths in [
            ("validation", x_val, y_val, val_paths),
            ("test", x_test, y_test, test_paths),
        ]:
            split_predictions = prediction_table(
                model_name=model_name,
                model=model,
                split_name=split_name,
                x_split=x_split,
                y_split=y_split,
                paths=paths,
            )
            split_predictions.to_csv(output_dir / f"{model_name}_{split_name}_predicoes.csv", index=False)
            prediction_rows.append(split_predictions)

        test_pred = model.predict(x_test)
        report = classification_report(y_test, test_pred, labels=LABEL_ORDER)
        (output_dir / f"{model_name}_classification_report.txt").write_text(
            report,
            encoding="utf-8",
        )
        save_confusion_matrix(
            y_test,
            test_pred,
            f"Matriz de confusao - {model_name}",
            output_dir / "matrizes_confusao" / f"{model_name}_test.png",
        )
        save_confusion_matrix(
            y_test,
            test_pred,
            f"Matriz de confusao normalizada - {model_name}",
            output_dir / "matrizes_confusao" / f"{model_name}_test_normalizada.png",
            normalize="true",
        )
        save_roc_curve(model, x_test, y_test, output_dir / "roc" / f"{model_name}.png")

        errors = pd.DataFrame(
            {
                "image_path": test_paths.values,
                "y_true": y_test.values,
                "y_pred": test_pred,
            }
        )
        errors = errors[errors["y_true"] != errors["y_pred"]]
        errors.to_csv(output_dir / f"{model_name}_test_errors.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics = metrics[
        [
            "model",
            "split",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_rotten",
            "recall_rotten",
            "f1_rotten",
            "best_params",
        ]
    ]
    metrics.to_csv(output_dir / "metricas_modelos.csv", index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            output_dir / "predicoes_modelos.csv",
            index=False,
        )
    save_metrics_barplot(metrics, output_dir / "metricas_modelos.png")
    return metrics


def save_metrics_barplot(metrics: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = metrics[metrics["split"].eq("test")].melt(
        id_vars=["model"],
        value_vars=["accuracy", "f1_macro", "recall_rotten", "f1_rotten"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right")
    plt.title("Comparacao dos modelos no conjunto de teste")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
