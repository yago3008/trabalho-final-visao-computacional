from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "figuras" / "pipeline_diagrama.png"


def add_box(ax, xy, text, width=2.35, height=0.95, color="#ffffff", edge="#315b7d"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#172033",
        weight="bold",
        linespacing=1.25,
    )
    return box


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.6,
        color="#667085",
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(arrow)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=180)
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#f6f8fb")
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(
        0.35,
        6.75,
        "Pipeline de inspecao visual automatica",
        fontsize=19,
        weight="bold",
        color="#101820",
        ha="left",
    )
    ax.text(
        0.35,
        6.35,
        "Classificacao fresh vs rotten com segmentacao, features manuais e modelos classicos",
        fontsize=11,
        color="#526070",
        ha="left",
    )

    boxes = {
        "dataset": add_box(
            ax,
            (0.45, 4.75),
            "Dataset\nfresh / rotten\namostra balanceada",
            color="#ffffff",
            edge="#315b7d",
        ),
        "prep": add_box(
            ax,
            (3.25, 4.75),
            "Pre-processamento\nresize 256x256\nsplit train/val/test",
            color="#ffffff",
            edge="#315b7d",
        ),
        "seg": add_box(
            ax,
            (6.05, 4.75),
            "Segmentacao\nHSV escolhido\nOtsu comparado",
            color="#eef7f1",
            edge="#1f8f5f",
        ),
        "features": add_box(
            ax,
            (8.85, 4.75),
            "Features manuais\nforma + Hu\ncor + textura",
            color="#fff7ed",
            edge="#b56a20",
        ),
        "analysis": add_box(
            ax,
            (3.25, 2.35),
            "Analise de features\nboxplots, correlacao\nPCA, SelectKBest",
            color="#ffffff",
            edge="#315b7d",
        ),
        "models": add_box(
            ax,
            (6.05, 2.35),
            "Classificadores\nLogReg, SVM RBF\nRandom Forest",
            color="#ffffff",
            edge="#315b7d",
        ),
        "eval": add_box(
            ax,
            (8.85, 2.35),
            "Avaliacao\nmetricas, matriz\nROC e erros",
            color="#fff1f1",
            edge="#b33b3b",
        ),
        "deliver": add_box(
            ax,
            (11.35, 2.35),
            "Entrega visual\nrelatorio\ndashboard Streamlit",
            color="#eef4ff",
            edge="#315b7d",
        ),
    }

    add_arrow(ax, (2.8, 5.22), (3.25, 5.22))
    add_arrow(ax, (5.6, 5.22), (6.05, 5.22))
    add_arrow(ax, (8.4, 5.22), (8.85, 5.22))
    add_arrow(ax, (10.02, 4.75), (4.42, 3.30))
    add_arrow(ax, (5.6, 2.82), (6.05, 2.82))
    add_arrow(ax, (8.4, 2.82), (8.85, 2.82))
    add_arrow(ax, (11.2, 2.82), (11.35, 2.82))

    ax.text(
        0.6,
        0.95,
        "Saidas geradas: X.csv, y.csv, graficos de features, matrizes de confusao, curvas ROC, exemplos de erros e XAI.",
        fontsize=10.5,
        color="#526070",
        ha="left",
    )
    ax.text(
        0.6,
        0.58,
        "Decisao da rodada demonstrativa: Random Forest por melhor F1 macro no teste e boa interpretabilidade.",
        fontsize=10.5,
        color="#526070",
        ha="left",
    )

    fig.tight_layout(pad=1.4)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
