# Checklist de requisitos do projeto

Legenda:

- `[x]` concluido no repositorio.
- `[ ]` pendente ou precisa de etapa manual/externa.
- `Parcial` significa que ha insumos prontos, mas falta transformar em entregavel final.

## 1. Requisitos tecnicos obrigatorios

| Status | Requisito | Evidencia | Observacao |
|---|---|---|---|
| [x] | Escolher um cenario de inspecao visual | `README.md`, `PLANO_EXECUCAO.md` | Cenario A: frutas `fresh` vs `rotten`. |
| [x] | Definir classes do problema | `data/processed/manifest.csv`, `y.csv` | Classes binarias: `fresh` e `rotten`. |
| [x] | Dataset com minimo de 100 imagens por classe | `data/processed/manifest.csv` | Amostra atual tem 140 imagens por classe: 80 treino, 20 validacao, 40 teste. |
| [x] | Buscar equilibrio entre classes | `data/processed/manifest.csv` | Todas as particoes estao balanceadas por classe. |
| [x] | Separar treino, validacao e teste | `data/processed/manifest.csv`, `y.csv` | Split atual: 160 treino, 40 validacao, 80 teste. |
| [x] | Fixar `random_state` para reprodutibilidade | `src/dataset.py`, `src/modeling.py` | Usado nos splits e modelos. |
| [x] | Segmentar ou isolar o objeto | `src/segmentation.py` | Segmentacao implementada no pipeline. |
| [x] | Comparar dois metodos de segmentacao | `outputs/figuras/segmentacao/` | Figuras com comparacao entre HSV e Otsu. |
| [x] | Gerar exemplos visuais da segmentacao | `outputs/figuras/segmentacao/` | 10 imagens comparativas geradas. |
| [x] | Extrair features manuais | `src/features.py`, `scripts/run_features.py`, `X.csv` | Features extraidas em tabela tabular. |
| [x] | Usar mascara nas features | `src/features.py` | Features calculadas sobre mascara segmentada. |
| [x] | Incluir features de forma | `X.csv` | Area, perimetro, circularidade, eccentricity, solidity e extent. |
| [x] | Incluir momentos de Hu | `X.csv` | `hu_1` a `hu_7`. |
| [x] | Incluir features de cor | `X.csv` | Medias e desvios RGB/HSV. |
| [x] | Incluir features de textura | `X.csv` | GLCM e LBP. |
| [x] | Gerar `X.csv` | `X.csv` | 280 linhas e 42 colunas na rodada atual. |
| [x] | Gerar `y.csv` | `y.csv` | Rotulos e splits alinhados ao `X.csv`. |
| [x] | Analisar features por classe | `outputs/figuras/features/boxplots.png` | Boxplots por classe gerados. |
| [x] | Calcular correlacao entre features | `outputs/figuras/features/correlacao.png` | Matriz de correlacao gerada. |
| [x] | Fazer PCA ou metodo equivalente | `outputs/figuras/features/pca.png` | PCA gerado para visualizacao. |
| [x] | Fazer selecao ou ranking de features | `outputs/figuras/features/select_k_best.csv` | Ranking SelectKBest gerado. |
| [x] | Treinar pelo menos dois classificadores classicos | `scripts/train_models.py`, `outputs/modelos/` | Foram treinados 3: Regressao Logistica, SVM RBF e Random Forest. |
| [x] | Comparar classificadores | `outputs/modelos/metricas_modelos.csv`, `outputs/modelos/metricas_modelos.png` | Tabela comparativa gerada. |
| [x] | Reportar acuracia | `outputs/modelos/metricas_modelos.csv` | Presente para validacao e teste. |
| [x] | Reportar precisao | `outputs/modelos/metricas_modelos.csv` | Presente para a classe `rotten`. |
| [x] | Reportar recall | `outputs/modelos/metricas_modelos.csv` | Presente para a classe `rotten`. |
| [x] | Reportar F1-score | `outputs/modelos/metricas_modelos.csv` | F1 macro e F1 da classe `rotten`. |
| [x] | Gerar matriz de confusao | `outputs/modelos/matrizes_confusao/` | Matrizes absolutas e normalizadas dos 3 modelos. |
| [x] | Gerar ROC ou matriz normalizada | `outputs/modelos/roc/`, `outputs/modelos/matrizes_confusao/` | ROC e matriz normalizada foram geradas. |
| [x] | Escolher melhor modelo com justificativa | `RESUMO_RESULTADOS.md`, `dashboard_app.py` | Random Forest escolhido pela melhor metrica de teste na rodada demonstrativa. |
| [x] | Fazer analise simples de erros com 5 a 10 imagens | `outputs/erros/random_forest/` | 10 imagens de erro separadas. |
| [x] | Escrever hipotese textual para cada erro no relatorio | `ANALISE_ERROS.md` | Hipoteses escritas para 10 imagens de erro. |
| [x] | Discutir limitacoes finais do dataset/features/modelo | `outputs/relatorio_tecnico.pdf`, `ROTEIRO_VIDEO.md` | Limitacoes consolidadas no relatorio e no roteiro. |

## 2. Estrutura exigida do repositorio

| Status | Requisito | Evidencia | Observacao |
|---|---|---|---|
| [x] | Pasta `notebooks/` | `notebooks/` | Pasta criada. |
| [x] | `01_segmentacao.ipynb` | `notebooks/01_segmentacao.ipynb` | Criado. |
| [x] | `02_features.ipynb` | `notebooks/02_features.ipynb` | Criado. |
| [x] | `03_classificacao.ipynb` | `notebooks/03_classificacao.ipynb` | Criado. |
| [x] | `04_cnn_xai_bonus.ipynb`, se houver bonus | `notebooks/04_cnn_xai_bonus.ipynb` | Criado como notebook de bonus/XAI. |
| [x] | Pasta `outputs/figuras` | `outputs/figuras/` | Contem segmentacao e features. |
| [x] | Matrizes de confusao | `outputs/modelos/matrizes_confusao/` | 6 imagens geradas. |
| [x] | Tabelas de metricas | `outputs/modelos/metricas_modelos.csv` | CSV gerado. |
| [x] | Graficos de features | `outputs/figuras/features/` | Boxplots, correlacao e PCA. |
| [x] | Imagens de erros | `outputs/erros/random_forest/` | 10 imagens copiadas. |
| [x] | `X.csv` na raiz | `X.csv` | Criado. |
| [x] | `y.csv` na raiz | `y.csv` | Criado. |
| [x] | `README.md` | `README.md` | Inclui metodologia e execucao. |
| [x] | `requirements.txt` ou `environment.yml` | `requirements.txt` | Dependencias listadas. |
| [x] | Codigo reproduzivel | `scripts/`, `src/`, `run_demo_pipeline.ps1` | Pipeline principal pode ser reexecutado. |
| [x] | README com execucao em ate 3 comandos | `README.md` | Ha caminho rapido com ambiente, pipeline e dashboard. |
| [ ] | Repositorio GitHub publico | `PUBLICACAO_GITHUB.md` | Guia pronto; falta criar o remoto, fazer push e postar o link. |

## 3. Relatorio tecnico em PDF

| Status | Requisito | Evidencia atual | O que falta |
|---|---|---|---|
| [x] | PDF de 6 a 10 paginas | `outputs/relatorio_tecnico.pdf` | Relatorio gerado com 9 paginas. |
| [x] | Problema e dataset | `README.md`, `DATASET.md`, `PLANO_EXECUCAO.md` | Conteudo base pronto. |
| [x] | Exemplos visuais do dataset no relatorio | `outputs/relatorio_tecnico.pdf`, `outputs/figuras/dataset_exemplos.png` | Inseridos no PDF. |
| [x] | Diagrama de blocos do pipeline | `outputs/figuras/pipeline_diagrama.png` | Figura criada e script reprodutivel em `scripts/create_pipeline_diagram.py`. |
| [x] | Segmentacao com dois metodos comparados | `outputs/figuras/segmentacao/` | Inserir e comentar no PDF. |
| [x] | Features extraidas com justificativa | `README.md`, `RESUMO_RESULTADOS.md`, `src/features.py` | Consolidar em texto final. |
| [x] | Analise de features com boxplots/metodos | `outputs/figuras/features/` | Inserir figuras no PDF. |
| [x] | Classificadores usados | `README.md`, `outputs/modelos/metricas_modelos.csv` | Consolidar em texto final. |
| [x] | Resultados e metricas | `RESUMO_RESULTADOS.md`, `outputs/modelos/` | Inserir tabela e figuras no PDF. |
| [x] | Analise de erros com hipotese | `ANALISE_ERROS.md`, `outputs/erros/random_forest/` | Hipoteses criadas para 10 exemplos. |
| [x] | Conclusao: modelo escolhido e motivo | `RESUMO_RESULTADOS.md` | Pode ser usado no PDF. |
| [x] | Bonus ou nivel avancado, se houver | `outputs/xai/`, `dashboard_app.py` | XAI e dashboard implementados. |

## 4. Video de apresentacao

| Status | Requisito | Evidencia | Observacao |
|---|---|---|---|
| [ ] | Video no YouTube como unlisted | `ROTEIRO_VIDEO.md` | Roteiro pronto; falta gravar e subir. |
| [ ] | Duracao maxima de 15 minutos | `ROTEIRO_VIDEO.md` | Planejado por blocos de tempo; falta conferir no video final. |
| [ ] | Problema escolhido em ate 3 min | `ROTEIRO_VIDEO.md` | Fala planejada; falta gravar. |
| [ ] | Tecnicas em ate 5 min | `ROTEIRO_VIDEO.md` | Fala planejada; falta gravar. |
| [ ] | Resultados em ate 4 min | `ROTEIRO_VIDEO.md`, dashboard | Fala planejada; falta gravar. |
| [ ] | Conclusao, limitacoes e melhorias em ate 3 min | `ROTEIRO_VIDEO.md` | Fala planejada; falta gravar. |
| [ ] | Postar links no Blackboard ate 07/06/2026 | - | Etapa manual externa. |

## 5. Bonus ou nivel avancado

| Status | Item bonus | Evidencia | Observacao |
|---|---|---|---|
| [x] | Permutation importance | `outputs/xai/permutation_importance.csv`, `outputs/xai/permutation_importance.png` | Implementado para Random Forest. |
| [x] | Interface simples em Streamlit ou Gradio | `dashboard_app.py`, `run_dashboard.ps1` | Dashboard Streamlit implementado e testado, incluindo galeria de imagens avaliadas com classe real e predita. |
| [ ] | SHAP | - | Nao implementado. Opcional. |
| [ ] | Ablation study formal por grupos | - | Nao implementado. Opcional, mas seria um bom proximo bonus. |
| [ ] | Transfer learning ou CNN pre-treinada | - | Nao implementado. Opcional; nao substitui pipeline classico. |
| [ ] | Grad-CAM | - | Depende de CNN pre-treinada, opcional. |
| [ ] | Analise de robustez com imagens novas | - | Opcional. |
| [ ] | Discussao de vies do dataset | - | Parcialmente planejada; falta texto final. |

## 6. Status geral

| Area | Status |
|---|---|
| Pipeline classico obrigatorio | [x] Completo |
| Graficos e avaliacoes | [x] Completo |
| Organizacao do repositorio local | [x] Completo |
| Dashboard bonus | [x] Completo |
| XAI bonus por permutation importance | [x] Completo |
| Relatorio PDF final | [x] Completo |
| Video final | [ ] Pendente |
| GitHub publico e Blackboard | [ ] Pendente |

## 7. Proximas acoes recomendadas

1. Gravar o video mostrando problema, dataset, tecnicas, dashboard, resultados e limitacoes.
2. Publicar o repositorio no GitHub e postar os links no Blackboard.
