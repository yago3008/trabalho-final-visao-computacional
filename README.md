# Sistema de inspecao visual automatica

Projeto final de Visao Computacional para classificacao de frutas dentro do padrao (`fresh`) versus fora do padrao (`rotten`) usando um pipeline classico de inspecao visual.

## Decisao tecnica

O projeto usa o cenario A, frutas frescas vs podres, porque ele e adequado para features classicas de cor, textura e forma. O modelo principal recomendado para discussao e o **Random Forest**, comparado com **SVM RBF** e **Regressao Logistica**.

## Estrutura do repositorio

```text
notebooks/
  01_segmentacao.ipynb
  02_features.ipynb
  03_classificacao.ipynb
  04_cnn_xai_bonus.ipynb
scripts/
  prepare_dataset.py
  run_segmentation_examples.py
  run_features.py
  analyze_features.py
  train_models.py
  copy_error_examples.py
  explain_model.py
  build_report.py
  create_pipeline_diagram.py
src/
  dataset.py
  segmentation.py
  features.py
  analysis.py
  modeling.py
outputs/
X.csv
y.csv
requirements.txt
```

## Execucao rapida

Crie o ambiente virtual e instale as dependencias:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se o comando `py` nao estiver disponivel no Windows, use o Python instalado diretamente:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Prepare uma amostra balanceada do dataset:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --train-per-class 500 --test-per-class 150
```

Execute o pipeline classico:

```powershell
.\.venv\Scripts\python.exe scripts/run_segmentation_examples.py
.\.venv\Scripts\python.exe scripts/run_features.py
.\.venv\Scripts\python.exe scripts/analyze_features.py
.\.venv\Scripts\python.exe scripts/train_models.py
.\.venv\Scripts\python.exe scripts/copy_error_examples.py
.\.venv\Scripts\python.exe scripts/explain_model.py
```

Ou execute a versao curta de demonstracao:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\run_demo_pipeline.ps1
```

O `Bypass` acima vale apenas para esse processo do PowerShell. Ele nao altera a politica global do Windows.

## Relatorio tecnico

Gerar o PDF do relatorio:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_report.py
```

Saida:

```text
outputs/relatorio_tecnico.pdf
```

## Dashboard web

O enunciado nao obriga web, mas cita **Streamlit ou Gradio** como bonus opcional. Este projeto inclui um dashboard em Streamlit para apresentar resultados, graficos, matrizes, ROC, segmentacao e erros.

Subir o dashboard:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\run_dashboard.ps1
```

Por padrao, o script tenta:

```text
http://127.0.0.1:11000
```

Se o Windows negar bind nessa porta, ele cai para:

```text
http://127.0.0.1:11001
```

No dashboard, o botao **Ver imagens classificadas** abre a galeria das imagens avaliadas pelo modelo selecionado. Nessa tela e possivel filtrar por modelo, split, classe real, classe prevista e status de acerto/erro. A galeria exibe a foto usada na avaliacao com sua classificacao, e a tabela detalhada mostra `y_true`, `y_pred`, probabilidades e confianca.

## Saidas principais

- `data/processed/manifest.csv`: imagens usadas no experimento.
- `X.csv`: tabela de features manuais.
- `y.csv`: rotulos e splits.
- `outputs/figuras/segmentacao`: comparacao visual HSV vs Otsu.
- `outputs/figuras/pipeline_diagrama.png`: diagrama de blocos do pipeline.
- `outputs/figuras/features`: boxplots, PCA, correlacao e SelectKBest.
- `outputs/modelos/metricas_modelos.csv`: tabela comparativa dos modelos.
- `outputs/modelos/predicoes_modelos.csv`: classificacao de cada imagem por modelo e split.
- `outputs/modelos/matrizes_confusao`: matrizes de confusao.
- `outputs/modelos/roc`: curvas ROC.
- `outputs/erros`: imagens classificadas incorretamente.
- `outputs/xai`: importancia por permutacao para explicar o modelo.
- `outputs/relatorio_tecnico.pdf`: relatorio tecnico de 6 a 10 paginas.
- `ROTEIRO_VIDEO.md`: guia de gravacao do video.

## Metodologia

1. Segmentacao por dois metodos: HSV e Otsu.
2. Extracao de features obrigatorias:
   - forma: area, perimetro, circularidade, eccentricity, solidity, extent;
   - inerciais: 7 momentos de Hu;
   - cor: medias e desvios RGB/HSV;
   - textura: GLCM e LBP.
3. Analise de features:
   - boxplots;
   - SelectKBest;
   - matriz de correlacao;
   - PCA.
4. Classificacao:
   - Regressao Logistica;
   - SVM com kernel RBF;
   - Random Forest.
5. Avaliacao:
   - acuracia;
   - precisao;
   - recall;
   - F1-score;
   - matriz de confusao;
   - curva ROC;
   - analise de erros.
6. Bonus:
   - permutation importance no Random Forest;
   - dashboard Streamlit.

## Resultados da rodada demonstrativa

| Modelo | Acuracia | F1 macro | Recall rotten | F1 rotten |
|---|---:|---:|---:|---:|
| Regressao Logistica | 80,0% | 80,0% | 77,5% | 79,5% |
| SVM RBF | 81,2% | 81,2% | 80,0% | 81,0% |
| Random Forest | 82,5% | 82,5% | 80,0% | 82,1% |

Modelo escolhido na rodada demonstrativa: **Random Forest**.

## Observacoes importantes

O script de preparo seleciona uma amostra balanceada e, por padrao, tenta manter uma unica variacao por imagem original para reduzir vazamento por aumentos artificiais.

Para uma demonstracao mais rapida:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --train-per-class 100 --test-per-class 40
```

## Problema do Python no Windows

Neste computador, `python` e `pip` apontavam para atalhos da Microsoft Store em `WindowsApps`, que falhavam com erro de sessao de logon. A solucao local foi usar diretamente o Python real instalado na maquina e criar a `.venv` a partir dele.

Em outro computador, prefira usar:

```powershell
py -3.11 -m venv .venv
```

ou:

```powershell
python -m venv .venv
```
