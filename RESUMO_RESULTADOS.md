# Resumo dos resultados da rodada demonstrativa

Rodada executada com:

- `--train-per-class 100`
- `--test-per-class 40`
- divisao final: 160 treino, 40 validacao, 80 teste
- segmentacao usada para features: `hsv`
- imagem redimensionada para `256x256`

## Features mais discriminativas

Ranking por `SelectKBest` no treino:

| Feature | Score | Interpretacao |
|---|---:|---|
| `glcm_homogeneity` | 88.34 | textura mais homogenea/irregular ajuda a separar frutas frescas e podres |
| `hsv_v_mean` | 62.18 | brilho medio muda com escurecimento e podridao |
| `rgb_r_mean` | 51.65 | canal vermelho captura diferencas de cor entre classes |
| `hsv_s_mean` | 48.57 | saturacao media muda com manchas e maturacao |
| `lbp_bin_2` | 32.87 | padroes locais de textura ajudam na classificacao |

Leitura didatica: o resultado e coerente com o problema. Frutas podres tendem a apresentar alteracoes de cor, brilho e textura.

## Comparacao dos modelos

| Modelo | Split | Accuracy | F1 macro | Recall rotten | F1 rotten |
|---|---|---:|---:|---:|---:|
| Regressao Logistica | validacao | 0.875 | 0.875 | 0.900 | 0.878 |
| Regressao Logistica | teste | 0.800 | 0.800 | 0.775 | 0.795 |
| SVM RBF | validacao | 0.875 | 0.874 | 0.800 | 0.865 |
| SVM RBF | teste | 0.812 | 0.812 | 0.800 | 0.810 |
| Random Forest | validacao | 0.800 | 0.799 | 0.850 | 0.810 |
| Random Forest | teste | 0.825 | 0.825 | 0.800 | 0.821 |

## Decisao do modelo

Modelo escolhido nesta rodada: **Random Forest**.

Justificativa: teve a melhor acuracia e melhor F1 macro no conjunto de teste, mantendo recall da classe `rotten` igual ao SVM. Alem disso, e mais interpretavel que o SVM por permitir importancia de variaveis e permutation importance.

## XAI por permutation importance

Principais features para o Random Forest no teste:

| Feature | Importancia media |
|---|---:|
| `hsv_s_mean` | 0.0439 |
| `lbp_bin_2` | 0.0249 |
| `rgb_r_mean` | 0.0113 |
| `lbp_bin_6` | 0.0112 |
| `rgb_b_std` | 0.0101 |

Leitura didatica: o modelo depende principalmente de saturacao, textura local e canais de cor, exatamente as familias esperadas para detectar deterioracao visual.

## Artefatos gerados

- `X.csv`
- `y.csv`
- `data/processed/manifest.csv`
- `outputs/figuras/segmentacao`
- `outputs/figuras/pipeline_diagrama.png`
- `outputs/figuras/features`
- `outputs/modelos/metricas_modelos.csv`
- `outputs/modelos/matrizes_confusao`
- `outputs/modelos/roc`
- `outputs/erros/random_forest`
- `ANALISE_ERROS.md`
- `outputs/xai/permutation_importance.csv`
- `outputs/xai/permutation_importance.png`
- `outputs/relatorio_tecnico.pdf`
- `ROTEIRO_VIDEO.md`
- `PUBLICACAO_GITHUB.md`
