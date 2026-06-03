from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "modelos"
FIGURES = OUTPUTS / "figuras"
ERRORS = OUTPUTS / "erros" / "random_forest"
XAI = OUTPUTS / "xai"
LABEL_ORDER = ["fresh", "rotten"]
VIEWS = ["Visao geral", "Imagens", "Segmentacao", "Features", "Modelos", "Erros", "XAI"]


st.set_page_config(
    page_title="Inspecao Visual Automatica",
    page_icon="CV",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f9fb;
            --ink: #18212f;
            --muted: #667085;
            --line: #d9e1ea;
            --fresh: #1f8f5f;
            --rotten: #b33b3b;
            --blue: #315b7d;
        }
        .stApp {
            background: linear-gradient(180deg, #fbfcfd 0%, #f4f7fa 100%);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: #101820;
        }
        [data-testid="stSidebar"] * {
            color: #eef5f7 !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] * {
            color: #101820 !important;
        }
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }
        h1 {
            font-size: 2.15rem !important;
            line-height: 1.12 !important;
            margin-bottom: .15rem !important;
        }
        h2 {
            font-size: 1.35rem !important;
            margin-top: 1.2rem !important;
        }
        h3 {
            font-size: 1.05rem !important;
        }
        .hero-note {
            color: var(--muted);
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .decision {
            border-left: 4px solid var(--fresh);
            background: #ffffff;
            padding: .8rem 1rem;
            border-radius: 8px;
            border-top: 1px solid var(--line);
            border-right: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .85rem 1rem;
            box-shadow: 0 10px 24px rgba(16, 24, 40, .05);
        }
        div[data-testid="stMetric"] label {
            color: #526070 !important;
            font-size: .82rem !important;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 800;
        }
        .small-caption {
            color: var(--muted);
            font-size: .84rem;
            margin-top: -.45rem;
            margin-bottom: .45rem;
        }
        .section-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(16, 24, 40, .04);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        .footer-note {
            color: var(--muted);
            border-top: 1px solid var(--line);
            padding-top: 1rem;
            margin-top: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_metrics() -> pd.DataFrame:
    path = MODELS / "metricas_modelos.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_select_k_best() -> pd.DataFrame:
    path = FIGURES / "features" / "select_k_best.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_permutation() -> pd.DataFrame:
    path = XAI / "permutation_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_manifest_summary() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    manifest = pd.read_csv(path)
    return (
        manifest.groupby(["split", "label"])
        .size()
        .reset_index(name="imagens")
        .sort_values(["split", "label"])
    )


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = MODELS / "predicoes_modelos.csv"
    if not path.exists():
        return pd.DataFrame()
    predictions = pd.read_csv(path)
    if "correct" in predictions.columns:
        predictions["correct"] = predictions["correct"].astype(bool)
    return predictions


def format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.1f}%"


def existing_images(folder: Path, pattern: str = "*.png", limit: int | None = None) -> list[Path]:
    if not folder.exists():
        return []
    files = sorted(folder.glob(pattern))
    return files[:limit] if limit else files


def image_grid(files: list[Path], columns: int = 3, caption_from_name: bool = True) -> None:
    if not files:
        st.info("Nenhuma imagem encontrada para esta secao.")
        return
    rows = [files[i : i + columns] for i in range(0, len(files), columns)]
    for row in rows:
        cols = st.columns(columns)
        for col, image_path in zip(cols, row):
            caption = image_path.stem.replace("_", " ") if caption_from_name else None
            col.image(str(image_path), caption=caption, use_container_width=True)


def prediction_gallery(rows: pd.DataFrame, limit: int, columns: int = 4) -> None:
    rows = rows.head(limit)
    if rows.empty:
        st.info("Nenhuma imagem encontrada com os filtros selecionados.")
        return

    for start in range(0, len(rows), columns):
        cols = st.columns(columns)
        for col, (_, row) in zip(cols, rows.iloc[start : start + columns].iterrows()):
            image_path = ROOT / str(row["image_path"])
            status = "ACERTO" if bool(row["correct"]) else "ERRO"
            confidence = row.get("confidence")
            confidence_text = f" | conf. {format_pct(confidence)}" if pd.notna(confidence) else ""
            caption = (
                f"{status} | real: {row['y_true']} | previsto: {row['y_pred']}"
                f"{confidence_text}"
            )
            if image_path.exists():
                col.image(str(image_path), caption=caption, use_container_width=True)
            else:
                col.warning(f"Imagem nao encontrada: {row['image_path']}")


def sidebar(metrics: pd.DataFrame) -> tuple[str, str]:
    st.sidebar.title("Inspecao Visual")
    st.sidebar.caption("Dashboard de resultados do pipeline classico")

    available_models = ["random_forest", "svm_rbf", "logistic_regression"]
    if not metrics.empty:
        available_models = metrics["model"].drop_duplicates().tolist()
    selected_model = st.sidebar.selectbox(
        "Modelo para inspecao",
        options=available_models,
        index=available_models.index("random_forest") if "random_forest" in available_models else 0,
    )
    selected_split = st.sidebar.selectbox("Split", ["test", "validation"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Requisitos cobertos**")
    st.sidebar.markdown(
        "- Dois metodos de segmentacao\n"
        "- Features manuais\n"
        "- Analise de features\n"
        "- Tres classificadores\n"
        "- Metricas e matriz de confusao\n"
        "- Analise de erros\n"
        "- Bonus XAI e dashboard"
    )
    return selected_model, selected_split


def header(metrics: pd.DataFrame, selected_model: str, selected_split: str) -> None:
    st.title("Inspecao Visual Automatica")
    st.markdown(
        "<div class='hero-note'>Classificacao de frutas fresh vs rotten com pipeline classico: "
        "segmentacao, features manuais, classificadores e avaliacao.</div>",
        unsafe_allow_html=True,
    )

    row = metrics[
        metrics["model"].eq(selected_model) & metrics["split"].eq(selected_split)
    ]
    current = row.iloc[0].to_dict() if not row.empty else {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acuracia", format_pct(current.get("accuracy")))
    col2.metric("F1 macro", format_pct(current.get("f1_macro")))
    col3.metric("Recall rotten", format_pct(current.get("recall_rotten")))
    col4.metric("F1 rotten", format_pct(current.get("f1_rotten")))

    b1, b2, _ = st.columns([0.28, 0.24, 0.48])
    if b1.button("Ver imagens classificadas", type="primary", use_container_width=True):
        st.session_state["active_view"] = "Imagens"
        st.rerun()
    if b2.button("Ver somente erros", type="primary", use_container_width=True):
        st.session_state["active_view"] = "Imagens"
        st.session_state["image_status_filter"] = "erros"
        st.rerun()


def overview_tab(metrics: pd.DataFrame, selected_model: str) -> None:
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.subheader("Comparacao dos modelos")
        if metrics.empty:
            st.warning("Arquivo de metricas ainda nao encontrado.")
        else:
            test_metrics = metrics[metrics["split"].eq("test")].copy()
            chart = test_metrics.melt(
                id_vars="model",
                value_vars=["accuracy", "f1_macro", "recall_rotten", "f1_rotten"],
                var_name="metrica",
                value_name="valor",
            )
            bars = (
                alt.Chart(chart)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("model:N", title="Modelo", axis=alt.Axis(labelAngle=-35)),
                    xOffset=alt.XOffset("metrica:N"),
                    y=alt.Y(
                        "valor:Q",
                        title="Valor",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%"),
                    ),
                    color=alt.Color(
                        "metrica:N",
                        title="Metrica",
                        scale=alt.Scale(
                            range=["#315b7d", "#6fa9d8", "#1f8f5f", "#b33b3b"]
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("model:N", title="Modelo"),
                        alt.Tooltip("metrica:N", title="Metrica"),
                        alt.Tooltip("valor:Q", title="Valor", format=".1%"),
                    ],
                )
                .properties(height=330)
            )
            st.altair_chart(bars, use_container_width=True)
            st.dataframe(
                test_metrics[
                    [
                        "model",
                        "accuracy",
                        "f1_macro",
                        "precision_rotten",
                        "recall_rotten",
                        "f1_rotten",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.subheader("Decisao de producao")
        st.markdown(
            "<div class='decision'><strong>Modelo escolhido na rodada demonstrativa: "
            "Random Forest.</strong><br><br>"
            "Ele obteve o melhor F1 macro no teste, manteve bom recall para a classe "
            "<code>rotten</code> e oferece explicabilidade por importancia de variaveis.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("### Amostra usada")
        summary = load_manifest_summary()
        if not summary.empty:
            st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Matriz de confusao e ROC")
    cm_col, roc_col = st.columns(2, gap="large")
    cm = MODELS / "matrizes_confusao" / f"{selected_model}_test_normalizada.png"
    roc = MODELS / "roc" / f"{selected_model}.png"
    with cm_col:
        if cm.exists():
            st.image(str(cm), caption="Matriz de confusao normalizada", use_container_width=True)
    with roc_col:
        if roc.exists():
            st.image(str(roc), caption="Curva ROC - classe rotten", use_container_width=True)


def segmentation_tab() -> None:
    st.subheader("Comparacao de segmentacao")
    st.markdown(
        "<div class='small-caption'>Cada figura compara imagem original, mascara por HSV e mascara por Otsu.</div>",
        unsafe_allow_html=True,
    )
    files = existing_images(FIGURES / "segmentacao", limit=12)
    image_grid(files, columns=2)


def features_tab() -> None:
    st.subheader("Analise de features")
    top = load_select_k_best()
    left, right = st.columns([.9, 1.1], gap="large")
    with left:
        st.markdown("### Ranking SelectKBest")
        if top.empty:
            st.info("Ranking ainda nao encontrado.")
        else:
            st.dataframe(top.head(12), use_container_width=True, hide_index=True)
    with right:
        st.markdown("### PCA")
        pca = FIGURES / "features" / "pca.png"
        if pca.exists():
            st.image(str(pca), use_container_width=True)

    st.markdown("### Graficos de distribuicao e correlacao")
    c1, c2 = st.columns(2, gap="large")
    boxplots = FIGURES / "features" / "boxplots.png"
    corr = FIGURES / "features" / "correlacao.png"
    if boxplots.exists():
        c1.image(str(boxplots), caption="Boxplots por classe", use_container_width=True)
    if corr.exists():
        c2.image(str(corr), caption="Correlacao entre features", use_container_width=True)


def models_tab(metrics: pd.DataFrame) -> None:
    st.subheader("Resultados dos classificadores")
    if metrics.empty:
        st.warning("Metricas nao encontradas.")
        return
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.markdown("### Matrizes de confusao")
    cm_files = existing_images(MODELS / "matrizes_confusao")
    image_grid(cm_files, columns=3)


def errors_tab() -> None:
    st.subheader("Analise de erros")
    st.markdown(
        "<div class='small-caption'>Exemplos em que o Random Forest errou no conjunto de teste.</div>",
        unsafe_allow_html=True,
    )
    errors_csv = MODELS / "random_forest_test_errors.csv"
    if errors_csv.exists():
        st.dataframe(pd.read_csv(errors_csv), use_container_width=True, hide_index=True)
    image_grid(existing_images(ERRORS, pattern="*.*", limit=10), columns=5)


def xai_tab() -> None:
    st.subheader("Explicabilidade por permutation importance")
    left, right = st.columns([.95, 1.05], gap="large")
    importance = load_permutation()
    with left:
        if importance.empty:
            st.info("Permutation importance ainda nao gerado.")
        else:
            st.dataframe(importance.head(20), use_container_width=True, hide_index=True)
    with right:
        image = XAI / "permutation_importance.png"
        if image.exists():
            st.image(str(image), use_container_width=True)


def images_tab(predictions: pd.DataFrame, selected_model: str, selected_split: str) -> None:
    st.subheader("Imagens utilizadas e classificacao")
    st.markdown(
        "<div class='small-caption'>Galeria das imagens avaliadas pelo modelo selecionado. "
        "Use os filtros para ver acertos, erros, classe real e classe prevista.</div>",
        unsafe_allow_html=True,
    )
    if predictions.empty:
        st.warning(
            "Arquivo de predicoes ainda nao encontrado. Execute `scripts/train_models.py` "
            "para gerar `outputs/modelos/predicoes_modelos.csv`."
        )
        return

    rows = predictions[
        predictions["model"].eq(selected_model) & predictions["split"].eq(selected_split)
    ].copy()
    if rows.empty:
        st.info("Nao ha predicoes para o modelo/split selecionado.")
        return

    total = len(rows)
    correct = int(rows["correct"].sum())
    errors = total - correct
    c1, c2, c3 = st.columns(3)
    c1.metric("Imagens avaliadas", total)
    c2.metric("Acertos", correct)
    c3.metric("Erros", errors)

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    true_filter = f1.selectbox("Classe real", ["todas"] + LABEL_ORDER, index=0)
    pred_filter = f2.selectbox("Classe prevista", ["todas"] + LABEL_ORDER, index=0)
    status_options = ["todos", "acertos", "erros"]
    status_default = st.session_state.pop("image_status_filter", "todos")
    status_index = status_options.index(status_default) if status_default in status_options else 0
    status_filter = f3.selectbox("Status", status_options, index=status_index)
    limit = f4.number_input("Max. imagens", min_value=4, max_value=80, value=24, step=4)

    if true_filter != "todas":
        rows = rows[rows["y_true"].eq(true_filter)]
    if pred_filter != "todas":
        rows = rows[rows["y_pred"].eq(pred_filter)]
    if status_filter == "acertos":
        rows = rows[rows["correct"]]
    elif status_filter == "erros":
        rows = rows[~rows["correct"]]

    rows = rows.sort_values(["correct", "y_true", "y_pred", "image_path"])
    st.markdown("### Galeria com classificacao")
    prediction_gallery(rows, limit=int(limit), columns=4)
    with st.expander("Tabela detalhada das predicoes", expanded=False):
        st.dataframe(
            rows[
                [
                    "image_path",
                    "y_true",
                    "y_pred",
                    "correct",
                    "confidence",
                    "proba_fresh",
                    "proba_rotten",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    inject_css()
    metrics = load_metrics()
    predictions = load_predictions()
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "Visao geral"
    selected_model, selected_split = sidebar(metrics)
    header(metrics, selected_model, selected_split)

    active_view = st.radio(
        "Navegacao",
        VIEWS,
        horizontal=True,
        label_visibility="collapsed",
        key="active_view",
    )

    if active_view == "Visao geral":
        overview_tab(metrics, selected_model)
    elif active_view == "Imagens":
        images_tab(predictions, selected_model, selected_split)
    elif active_view == "Segmentacao":
        segmentation_tab()
    elif active_view == "Features":
        features_tab()
    elif active_view == "Modelos":
        models_tab(metrics)
    elif active_view == "Erros":
        errors_tab()
    elif active_view == "XAI":
        xai_tab()

    st.markdown(
        "<div class='footer-note'>Dashboard de apoio didatico. Os resultados foram gerados "
        "por scripts reprodutiveis no repositorio.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
