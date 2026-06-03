from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "relatorio_tecnico.pdf"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.HexColor("#101820"),
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom",
            parent=base["Normal"],
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#526070"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "HeadingCustom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            spaceBefore=8,
            spaceAfter=8,
            textColor=colors.HexColor("#172033"),
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=6,
            spaceAfter=4,
            textColor=colors.HexColor("#315b7d"),
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontSize=9.4,
            leading=12.2,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            textColor=colors.HexColor("#1d2939"),
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["BodyText"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#667085"),
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "SmallCustom",
            parent=base["BodyText"],
            fontSize=8.2,
            leading=10.3,
            textColor=colors.HexColor("#344054"),
        ),
    }


def p(text: str, style: ParagraphStyle):
    return Paragraph(text.replace("\n", "<br/>"), style)


def image_flowable(path: Path, max_width: float, max_height: float):
    if not path.exists():
        return p(f"Figura nao encontrada: {path}", styles()["caption"])
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def table_flowable(data: list[list[str]], col_widths: list[float] | None = None, font_size: float = 8):
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172033")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d5dd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def dataset_grid() -> Path:
    out = ROOT / "outputs" / "figuras" / "dataset_exemplos.png"
    if out.exists():
        return out

    from PIL import ImageDraw, ImageFont

    manifest = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv")
    samples = []
    for label in ["fresh", "rotten"]:
        label_rows = manifest[manifest["label"].eq(label)].head(4)
        samples.extend(label_rows.to_dict("records"))

    thumb_w, thumb_h, label_h = 180, 150, 32
    sheet = PILImage.new("RGB", (4 * thumb_w, 2 * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for idx, record in enumerate(samples):
        image = PILImage.open(ROOT / record["image_path"]).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), PILImage.Resampling.LANCZOS)
        x = (idx % 4) * thumb_w
        y = (idx // 4) * (thumb_h + label_h)
        bg = PILImage.new("RGB", (thumb_w, thumb_h), (246, 248, 251))
        bg.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        sheet.paste(bg, (x, y))
        draw.text((x + 8, y + thumb_h + 7), record["label"], fill=(16, 24, 32), font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def format_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def build_story() -> list:
    s = styles()
    story = []
    width = A4[0] - 3.0 * cm

    metrics = pd.read_csv(ROOT / "outputs" / "modelos" / "metricas_modelos.csv")
    manifest = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv")
    select_k = pd.read_csv(ROOT / "outputs" / "figuras" / "features" / "select_k_best.csv")
    xai = pd.read_csv(ROOT / "outputs" / "xai" / "permutation_importance.csv")

    story.append(p("Sistema de inspecao visual automatica", s["title"]))
    story.append(
        p(
            "Projeto final de Visao Computacional: classificacao de frutas fresh vs rotten usando pipeline classico com segmentacao, features manuais, classificadores e avaliacao.",
            s["subtitle"],
        )
    )
    story.append(p("1. Problema e dataset", s["h1"]))
    story.append(
        p(
            "O problema escolhido foi o cenario de frutas frescas versus frutas podres. A tarefa representa uma inspecao visual binaria: itens dentro do padrao sao classificados como fresh, e itens fora do padrao visual sao classificados como rotten. A escolha e adequada para um pipeline classico porque mudancas de deterioracao costumam aparecer em cor, brilho, textura e formato.",
            s["body"],
        )
    )
    counts = manifest.groupby(["split", "label"]).size().reset_index(name="imagens")
    data = [["Split", "Classe", "Imagens"]] + counts.astype(str).values.tolist()
    story.append(table_flowable(data, [3.5 * cm, 3.5 * cm, 3.0 * cm]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(image_flowable(dataset_grid(), width, 7.0 * cm))
    story.append(p("Figura 1. Exemplos do dataset processado nas classes fresh e rotten.", s["caption"]))
    story.append(PageBreak())

    story.append(p("2. Pipeline proposto", s["h1"]))
    story.append(
        p(
            "O pipeline separa as etapas de preparacao, segmentacao, extracao de atributos, analise de features, classificacao e avaliacao. Essa organizacao permite reproduzir cada etapa e discutir onde os acertos e falhas acontecem.",
            s["body"],
        )
    )
    story.append(image_flowable(ROOT / "outputs" / "figuras" / "pipeline_diagrama.png", width, 13.5 * cm))
    story.append(p("Figura 2. Diagrama de blocos do pipeline classico implementado.", s["caption"]))
    story.append(p("3. Segmentacao ou isolamento do objeto", s["h1"]))
    story.append(
        p(
            "Foram comparados dois metodos: limiarizacao em HSV e limiarizacao por Otsu. O HSV foi usado como metodo principal para extracao de features por lidar melhor com diferencas de cor e fundo no subconjunto analisado. Otsu foi mantido como comparacao visual para documentar acertos e falhas.",
            s["body"],
        )
    )
    seg = sorted((ROOT / "outputs" / "figuras" / "segmentacao").glob("*.png"))[0]
    story.append(image_flowable(seg, width, 9.0 * cm))
    story.append(p("Figura 3. Exemplo de comparacao entre segmentacao HSV e Otsu.", s["caption"]))
    story.append(PageBreak())

    story.append(p("4. Features extraidas", s["h1"]))
    story.append(
        p(
            "O vetor de entrada combina familias de features vistas em aula: forma, momentos de Hu, cor e textura. As features de forma resumem area, perimetro, circularidade, eccentricidade, solidez e extent. Os momentos de Hu capturam propriedades inerciais da mascara. As features de cor usam medias e desvios em RGB e HSV. As features de textura usam GLCM e LBP.",
            s["body"],
        )
    )
    features_data = [
        ["Familia", "Exemplos", "Justificativa"],
        ["Forma", "area, perimetro, circularidade", "Defeitos e recortes podem alterar contorno."],
        ["Hu", "hu_1 a hu_7", "Resumo invariante da geometria da mascara."],
        ["Cor", "RGB/HSV media e desvio", "Podridao altera brilho, saturacao e tonalidade."],
        ["Textura", "GLCM, LBP", "Manchas e enrugamento mudam padroes locais."],
    ]
    story.append(table_flowable(features_data, [3.0 * cm, 5.5 * cm, 7.2 * cm], font_size=7.8))
    story.append(p("5. Selecao e analise de features", s["h1"]))
    top_data = [["Feature", "Score"]] + [
        [row["feature"], f"{row['score']:.2f}"] for _, row in select_k.head(8).iterrows()
    ]
    story.append(p("O SelectKBest destacou textura e cor como principais familias discriminativas.", s["body"]))
    story.append(table_flowable(top_data, [7.0 * cm, 3.0 * cm]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(image_flowable(ROOT / "outputs" / "figuras" / "features" / "boxplots.png", width, 9.0 * cm))
    story.append(p("Figura 4. Boxplots das features por classe.", s["caption"]))
    story.append(PageBreak())

    story.append(p("Analise visual complementar", s["h1"]))
    story.append(
        p(
            "A matriz de correlacao ajuda a identificar redundancias entre atributos, enquanto o PCA oferece uma visao compacta da separacao entre classes. Esses graficos nao substituem a avaliacao supervisionada, mas apoiam a interpretacao do comportamento do vetor de features.",
            s["body"],
        )
    )
    story.append(image_flowable(ROOT / "outputs" / "figuras" / "features" / "correlacao.png", width, 10.0 * cm))
    story.append(p("Figura 5. Correlacao entre features.", s["caption"]))
    story.append(image_flowable(ROOT / "outputs" / "figuras" / "features" / "pca.png", width, 8.2 * cm))
    story.append(p("Figura 6. Visualizacao PCA do conjunto de features.", s["caption"]))
    story.append(PageBreak())

    story.append(p("6. Classificadores usados", s["h1"]))
    story.append(
        p(
            "Foram comparados tres modelos classicos sobre a mesma tabela de features: Regressao Logistica como baseline linear, SVM com kernel RBF para fronteira nao linear e Random Forest para capturar interacoes entre atributos e permitir interpretabilidade por importancia de variaveis.",
            s["body"],
        )
    )
    test = metrics[metrics["split"].eq("test")].copy()
    model_rows = [["Modelo", "Acuracia", "F1 macro", "Precisao rotten", "Recall rotten", "F1 rotten"]]
    for _, row in test.iterrows():
        model_rows.append(
            [
                row["model"],
                format_pct(row["accuracy"]),
                format_pct(row["f1_macro"]),
                format_pct(row["precision_rotten"]),
                format_pct(row["recall_rotten"]),
                format_pct(row["f1_rotten"]),
            ]
        )
    story.append(table_flowable(model_rows, [3.5 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm, 2.5 * cm, 2.2 * cm], font_size=7.5))
    story.append(Spacer(1, 0.2 * cm))
    story.append(image_flowable(ROOT / "outputs" / "modelos" / "metricas_modelos.png", width, 8.5 * cm))
    story.append(p("Figura 7. Comparacao de metricas entre os classificadores.", s["caption"]))
    story.append(PageBreak())

    story.append(p("7. Resultados e metricas", s["h1"]))
    story.append(
        p(
            "O melhor resultado no teste foi obtido pelo Random Forest, com acuracia de 82,5%, F1 macro de 82,5% e F1 da classe rotten de 82,1%. Como a classe rotten representa defeito, o recall dessa classe tambem foi observado; Random Forest e SVM atingiram 80,0% nessa metrica.",
            s["body"],
        )
    )
    story.append(image_flowable(ROOT / "outputs" / "modelos" / "matrizes_confusao" / "random_forest_test_normalizada.png", width / 2, 8.0 * cm))
    story.append(p("Figura 8. Matriz de confusao normalizada do Random Forest.", s["caption"]))
    story.append(image_flowable(ROOT / "outputs" / "modelos" / "roc" / "random_forest.png", width / 2, 8.0 * cm))
    story.append(p("Figura 9. Curva ROC do Random Forest para a classe rotten.", s["caption"]))
    story.append(PageBreak())

    story.append(p("8. Analise de erros", s["h1"]))
    story.append(
        p(
            "A analise de erros mostra falsos positivos e falsos negativos. Falsos positivos ocorreram em frutas frescas com brilho, textura natural intensa, sombras ou fundo residual. Falsos negativos ocorreram em frutas podres quando a deterioracao era localizada, quando a cor global ainda era semelhante a frutas aceitaveis ou quando fundo/rotacao afetaram a mascara.",
            s["body"],
        )
    )
    story.append(image_flowable(ROOT / "outputs" / "figuras" / "analise_erros" / "erros_random_forest_prancha.png", width, 8.5 * cm))
    story.append(p("Figura 10. Dez exemplos de erro do Random Forest com rotulo real e predito.", s["caption"]))
    error_data = [
        ["Tipo", "Hipotese resumida"],
        ["Fresh -> rotten", "Textura de casca, sombras, brilho e tons escuros podem parecer deterioracao."],
        ["Rotten -> fresh", "Defeito localizado ou cor global ainda aceitavel pode mascarar sinais de podridao."],
        ["Segmentacao", "Bordas pretas, fundo residual e rotacao alteram forma, cor e textura."],
    ]
    story.append(table_flowable(error_data, [4.0 * cm, 11.8 * cm], font_size=8))
    story.append(PageBreak())

    story.append(p("9. Conclusao e limitacoes", s["h1"]))
    story.append(
        p(
            "Para a rodada demonstrativa, o modelo recomendado para producao seria o Random Forest. Ele apresentou o melhor F1 macro no teste, manteve recall competitivo para a classe rotten e permite explicacao por permutation importance. Mesmo assim, a solucao ainda tem limitacoes: depende da qualidade da segmentacao, usa estatisticas globais que podem diluir defeitos pequenos e foi avaliada em uma amostra controlada do dataset.",
            s["body"],
        )
    )
    story.append(p("10. Bonus e nivel avancado", s["h1"]))
    story.append(
        p(
            "Como bonus, foi implementada explicabilidade por permutation importance e uma interface Streamlit para explorar metricas, figuras, matrizes, erros e XAI. As principais features por permutation importance reforcam a coerencia tecnica: saturacao, textura LBP e canais de cor aparecem entre os atributos mais influentes.",
            s["body"],
        )
    )
    xai_data = [["Feature", "Importancia media"]] + [
        [row["feature"], f"{row['importance_mean']:.4f}"] for _, row in xai.head(8).iterrows()
    ]
    story.append(table_flowable(xai_data, [7.0 * cm, 4.0 * cm]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(image_flowable(ROOT / "outputs" / "xai" / "permutation_importance.png", width, 8.0 * cm))
    story.append(p("Figura 11. Permutation importance do Random Forest.", s["caption"]))
    story.append(PageBreak())

    story.append(p("Referencias de execucao e reprodutibilidade", s["h1"]))
    story.append(
        p(
            "O repositorio contem notebooks, scripts e modulos organizados por etapa. O pipeline principal pode ser reproduzido pelos comandos do README. As principais saidas sao X.csv, y.csv, manifest.csv, graficos de segmentacao, graficos de features, metricas, matrizes de confusao, curvas ROC, exemplos de erros e dashboard Streamlit.",
            s["body"],
        )
    )
    repo_data = [
        ["Artefato", "Papel"],
        ["X.csv / y.csv", "Tabela de features e rotulos."],
        ["src/", "Implementacao reutilizavel de dataset, segmentacao, features, analise e modelos."],
        ["scripts/", "Execucao reprodutivel de cada etapa."],
        ["outputs/", "Figuras, metricas e evidencias usadas no relatorio."],
        ["dashboard_app.py", "Interface Streamlit para apresentacao dos resultados."],
    ]
    story.append(table_flowable(repo_data, [5.0 * cm, 10.8 * cm], font_size=8))

    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(1.5 * cm, 1.0 * cm, "Projeto Final - Visao Computacional")
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.45 * cm,
        title="Relatorio tecnico - Inspecao Visual Automatica",
    )
    story = build_story()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
