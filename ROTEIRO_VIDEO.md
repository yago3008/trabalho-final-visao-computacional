# Roteiro para video de apresentacao

Duracao maxima exigida: 15 minutos.

Objetivo do video: apresentar problema, dataset, tecnicas, resultados, erros, conclusao e limitacoes. O foco nao deve ser explicar o codigo linha por linha.

## 0. Preparacao antes de gravar

1. Abrir o dashboard:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\run_dashboard.ps1
```

2. Abrir o relatorio em PDF:

```text
outputs/relatorio_tecnico.pdf
```

3. Deixar abertas as principais figuras:

- `outputs/figuras/pipeline_diagrama.png`
- `outputs/figuras/segmentacao/`
- `outputs/figuras/features/boxplots.png`
- `outputs/modelos/matrizes_confusao/random_forest_test_normalizada.png`
- `outputs/modelos/roc/random_forest.png`
- `outputs/figuras/analise_erros/erros_random_forest_prancha.png`

## 1. Problema escolhido e dataset - ate 3 min

Fala sugerida:

> O projeto resolve uma tarefa de inspecao visual automatica para classificar frutas em duas classes: fresh, quando estao dentro do padrao visual, e rotten, quando apresentam deterioracao. Escolhemos esse cenario porque ele combina bem com features classicas de cor, textura e forma: frutas podres tendem a apresentar manchas, escurecimento, mudanca de saturacao e irregularidades de textura.

Pontos para mostrar:

- Exemplos de imagens fresh e rotten.
- Balanceamento da amostra: 80 treino, 20 validacao e 40 teste por classe.
- `X.csv`, `y.csv` e `data/processed/manifest.csv`.

## 2. Tecnicas utilizadas - ate 5 min

Fala sugerida:

> O pipeline foi organizado em etapas: preparo do dataset, segmentacao, extracao de features, analise de features, treinamento de classificadores e avaliacao. Para segmentacao, comparamos HSV e Otsu. Para extracao de features, usamos forma, momentos de Hu, cor em RGB/HSV e textura com GLCM e LBP.

Pontos para mostrar:

- Diagrama em `outputs/figuras/pipeline_diagrama.png`.
- Comparacao HSV vs Otsu.
- Explicar que HSV foi usado como metodo principal para features.
- Families de features:
  - forma: area, perimetro, circularidade, eccentricity, solidity, extent;
  - momentos de Hu: `hu_1` a `hu_7`;
  - cor: medias e desvios RGB/HSV;
  - textura: GLCM e LBP.
- Analise de features:
  - boxplots;
  - correlacao;
  - PCA;
  - SelectKBest.

## 3. Resultados obtidos - ate 4 min

Fala sugerida:

> Foram comparados tres classificadores classicos: Regressao Logistica, SVM com kernel RBF e Random Forest. O Random Forest foi escolhido como melhor modelo da rodada demonstrativa porque teve a melhor acuracia e o melhor F1 macro no teste, mantendo bom recall para a classe rotten e oferecendo interpretabilidade por importancia de variaveis.

Metricas principais do teste:

| Modelo | Acuracia | F1 macro | Recall rotten | F1 rotten |
|---|---:|---:|---:|---:|
| Regressao Logistica | 80,0% | 80,0% | 77,5% | 79,5% |
| SVM RBF | 81,2% | 81,2% | 80,0% | 81,0% |
| Random Forest | 82,5% | 82,5% | 80,0% | 82,1% |

Pontos para mostrar:

- Tabela comparativa no dashboard.
- Matriz de confusao normalizada.
- Curva ROC.
- Prancha de erros.
- Permutation importance.

## 4. Conclusao, limitacoes e melhorias - ate 3 min

Fala sugerida:

> A solucao cumpre o objetivo do pipeline classico: segmenta o objeto, extrai features manuais, compara classificadores e avalia os resultados. O modelo escolhido seria o Random Forest, mas ainda existem limitacoes. O pipeline depende da qualidade da segmentacao; defeitos pequenos podem ser diluidos por medias globais; brilho, sombras, fundo preto e rotacao podem confundir o modelo. Melhorias futuras incluem features locais por regioes, refinamento da segmentacao, ablation study por grupos de features e teste com imagens novas fora do dataset.

Pontos para fechar:

- Nao prometer 100% de acerto.
- Explicar que os erros sao coerentes com as limitacoes das features classicas.
- Mencionar bonus implementado:
  - Streamlit;
  - permutation importance.

## Checklist de gravacao

- [ ] Video com ate 15 minutos.
- [ ] Mostra problema e dataset.
- [ ] Mostra pipeline e segmentacao.
- [ ] Mostra features e analise.
- [ ] Mostra classificadores e metricas.
- [ ] Mostra matriz de confusao e ROC.
- [ ] Mostra exemplos de erros.
- [ ] Fecha com conclusao, limitacoes e melhorias.
- [ ] Enviado ao YouTube como unlisted.
- [ ] Link pronto para Blackboard.
