# Plano de execucao - Trabalho final de Visao Computacional

Este documento resume o plano tecnico para o projeto de inspecao visual automatica. Ele foi feito como guia de execucao e registro de decisoes, nao como relatorio final pronto.

## 1. Leitura do enunciado

O trabalho pede um pipeline classico de visao computacional:

1. aquisicao ou escolha do dataset;
2. pre-processamento;
3. segmentacao ou isolamento do objeto;
4. extracao de features manuais;
5. montagem da tabela `X` e vetor `y`;
6. treino de pelo menos dois classificadores classicos;
7. avaliacao com metricas e matriz de confusao;
8. analise critica dos resultados.

Entregas esperadas:

- repositorio GitHub publico;
- relatorio tecnico em PDF, de 6 a 10 paginas;
- video de ate 15 minutos;
- `README.md`, `requirements.txt`, `X.csv`, `y.csv` e figuras/resultados em `outputs/`.

## 2. Comparacao dos 3 cenarios

| Cenario | Vantagens | Riscos | Viabilidade |
|---|---|---|---|
| A - Frutas | Datasets abundantes, problema visual intuitivo, cor e textura ajudam bastante, segmentacao tende a ser simples em fundo controlado. | Pode haver variacao de iluminacao e fundo se o dataset nao for padronizado. | Alta |
| B - Graos de cafe | Tema forte para inspecao industrial, defeitos reais e boa relacao com textura/cor. | Objetos pequenos, defeitos sutis, classes mais dificeis, segmentacao e rotulagem podem ser mais trabalhosas. | Media |
| C - Comprimidos | Fundo controlado e objetos simples; bom para forma, cor e defeitos visuais. | Dataset pode ser mais limitado; defeitos podem ser pequenos e exigir boa resolucao. | Media-alta |

## 3. Decisao principal: melhor cenario

**Escolha recomendada: Cenario A - frutas frescas vs podres.**

Breve explicacao: e o melhor equilibrio entre aderencia ao enunciado, facilidade de obter dados, segmentacao viavel e features classicas bem justificaveis. Cor e textura tendem a separar bem frutas frescas de frutas podres, e o problema permite uma avaliacao clara com metricas como acuracia, precisao, recall, F1 e matriz de confusao.

## 4. Escopo recomendado

**Problema recomendado:** classificacao binaria entre `fresh` e `rotten`.

Breve explicacao: o problema binario reduz complexidade, facilita balanceamento das classes e permite usar curva ROC alem da matriz de confusao. Tambem evita que o projeto fique grande demais para o prazo.

**Dataset recomendado:** Fruit Quality Detection ou subconjunto equivalente de frutas frescas e podres.

Breve explicacao: esse tipo de dataset esta alinhado diretamente ao cenario A e costuma ter imagens ja organizadas por classe. A meta deve ser usar no minimo 100 imagens por classe, idealmente 200 ou mais por classe.

**Classes iniciais:** `fresh` e `rotten`.

Breve explicacao: com duas classes, a analise fica mais objetiva e a avaliacao mais confiavel. Se houver tempo, a equipe pode separar por fruta, como maca, banana e laranja, mas isso deve ser tratado como extensao.

## 5. Nivel tecnico alvo

**Alvo recomendado: nivel intermediario.**

Breve explicacao: o nivel intermediario cobre todo o obrigatorio e adiciona analise de features, PCA, SelectKBest, importancia de variaveis e validacao cruzada. Isso aumenta a qualidade do trabalho sem depender de CNN ou XAI, que podem consumir muito tempo.

**Bonus opcional:** comparar com MobileNetV2 ou ResNet50 apenas se o pipeline classico estiver completo.

Breve explicacao: transfer learning pode melhorar desempenho, mas nao substitui a parte obrigatoria. Deve entrar somente como comparacao extra.

## 6. Estrategia de dados

**Balanceamento:** usar a mesma quantidade de imagens por classe.

Breve explicacao: evita que a acuracia fique artificialmente alta por causa de uma classe majoritaria.

**Divisao dos dados:** `train`, `validation` e `test` estratificados.

Breve explicacao: o treino ajusta o modelo, a validacao ajuda na escolha de parametros e o teste fica reservado para a avaliacao final.

**Reprodutibilidade:** fixar `random_state`.

Breve explicacao: resultados reprodutiveis facilitam comparacao entre modelos e evitam variacao aleatoria entre execucoes.

## 7. Estrategia de pre-processamento

**Padronizacao de tamanho:** redimensionar imagens para uma resolucao fixa.

Breve explicacao: simplifica o processamento e evita que features dependam diretamente do tamanho original da imagem.

**Realce e suavizacao:** aplicar conversao de cor, suavizacao leve e ajustes simples quando necessario.

Breve explicacao: reduz ruido sem destruir manchas, bordas e textura, que sao importantes para o problema.

**Uso de mascara:** calcular features apenas sobre a regiao da fruta.

Breve explicacao: evita que o fundo influencie medias de cor, textura e forma.

## 8. Estrategia de segmentacao

**Metodo 1:** segmentacao por limiar em HSV ou RGB, usando diferenca entre fruta e fundo.

Breve explicacao: e simples, interpretavel e combina bem com imagens de frutas sobre fundo controlado.

**Metodo 2:** Otsu ou segmentacao por limiar no canal de intensidade/saturacao, com pos-processamento morfologico.

Breve explicacao: permite comparar uma segunda abordagem classica, como o enunciado pede, e documentar acertos e falhas.

**Criterio de escolha:** selecionar o metodo que melhor isola a fruta com menos ruido de fundo.

Breve explicacao: a segmentacao nao precisa ser perfeita, mas precisa ser coerente e permitir extrair features confiaveis.

## 9. Features manuais

**Forma:** area, perimetro, circularidade, eccentricity, solidity e extent.

Breve explicacao: frutas podres ou danificadas podem apresentar deformacoes, rachaduras ou contornos menos regulares.

**Momentos de Hu:** 7 momentos em escala logaritmica.

Breve explicacao: atendem ao requisito de features inerciais e resumem propriedades globais da forma.

**Cor:** medias RGB/HSV e, se possivel, histogramas de matiz.

Breve explicacao: podridao, maturacao irregular e manchas aparecem principalmente como alteracoes de cor.

**Textura:** GLCM ou LBP.

Breve explicacao: manchas, mofo e rugosidade tendem a alterar textura; por isso, essa familia deve ser uma das mais importantes.

## 10. Analise e selecao de features

**Boxplots por classe.**

Breve explicacao: ajudam a ver quais features separam visualmente `fresh` e `rotten`.

**Matriz de correlacao.**

Breve explicacao: identifica features redundantes e ajuda a explicar escolhas.

**PCA para visualizacao.**

Breve explicacao: mostra se as classes formam grupos separados em baixa dimensao.

**SelectKBest.**

Breve explicacao: oferece uma selecao simples e justificavel das features mais discriminativas.

**Importancia do Random Forest.**

Breve explicacao: ajuda a interpretar quais variaveis mais influenciam o classificador.

## 11. Classificadores recomendados

**Modelo 1 - Regressao Logistica.**

Breve explicacao: serve como baseline forte, simples e interpretavel. Exige escalonamento das features.

**Modelo 2 - SVM.**

Breve explicacao: costuma funcionar bem com datasets pequenos ou medios e fronteiras nao lineares. Exige escalonamento e ajuste de parametros.

**Modelo 3 - Random Forest.**

Breve explicacao: lida bem com features de natureza diferente, captura relacoes nao lineares e fornece importancia de variaveis.

## 12. Melhor modelo para priorizar

**Modelo recomendado para desenvolvimento principal: Random Forest.**

Breve explicacao: entre os tres modelos propostos, Random Forest e o melhor ponto de partida porque e robusto, interpretavel, nao depende tanto de escalonamento quanto SVM e Regressao Logistica, e combina bem com features manuais de cor, textura e forma. Ainda assim, a decisao final deve ser confirmada pelas metricas no conjunto de teste.

**Modelo de comparacao obrigatoria:** SVM.

Breve explicacao: SVM e uma comparacao forte para features manuais e pode superar Random Forest dependendo da separabilidade dos dados.

**Modelo baseline:** Regressao Logistica.

Breve explicacao: ajuda a mostrar se modelos mais complexos realmente trazem ganho.

## 13. Metricas de avaliacao

**Metricas obrigatorias:** acuracia, precisao, recall, F1-score e matriz de confusao.

Breve explicacao: acuracia sozinha pode enganar; F1 e recall sao especialmente importantes se a classe `rotten` for tratada como defeito.

**Metrica principal para decisao:** F1-score da classe defeituosa ou F1 macro.

Breve explicacao: em inspecao visual, deixar passar item defeituoso costuma ser pior do que descartar um item bom.

**Visualizacao:** matriz de confusao normalizada e curva ROC.

Breve explicacao: ajudam a explicar onde o sistema erra e qual e o compromisso entre falsos positivos e falsos negativos.

## 14. Analise de erros

**Selecionar 5 a 10 erros do melhor modelo.**

Breve explicacao: o enunciado pede analise critica; exemplos visuais ajudam a entender se o erro veio de iluminacao, segmentacao, fundo, fruta ambigua ou feature insuficiente.

**Separar falsos positivos e falsos negativos.**

Breve explicacao: em inspecao de defeitos, os dois tipos de erro tem impactos diferentes.

## 15. Organizacao sugerida do repositorio

```text
notebooks/
  01_segmentacao.ipynb
  02_features.ipynb
  03_classificacao.ipynb
  04_cnn_xai_bonus.ipynb
outputs/
  figuras/
  matrizes_confusao/
  metricas/
  erros/
data/
  raw/
  processed/
X.csv
y.csv
README.md
requirements.txt
PLANO_EXECUCAO.md
```

Breve explicacao: essa estrutura segue o enunciado e facilita reproducao, revisao e organizacao dos resultados.

## 16. Cronograma de execucao

| Etapa | Objetivo |
|---|---|
| Dia 1 | Escolher dataset, baixar/organizar imagens, definir classes e balanceamento. |
| Dia 2 | Testar dois metodos de segmentacao e salvar exemplos visuais. |
| Dia 3 | Extrair features e gerar `X.csv` e `y.csv`. |
| Dia 4 | Fazer analise de features: boxplots, correlacao, PCA e SelectKBest. |
| Dia 5 | Treinar Regressao Logistica, SVM e Random Forest; comparar metricas. |
| Dia 6 | Fazer analise de erros, organizar outputs e escrever conclusoes tecnicas. |
| Dia 7 | Revisar repositorio, README, relatorio e apresentacao. |

## 17. Riscos e mitigacoes

**Risco:** dataset com fundo muito variado.

Mitigacao: escolher subconjunto com fundo controlado ou usar recorte/mascara antes das features.

**Risco:** classes desbalanceadas.

Mitigacao: limitar todas as classes ao mesmo numero de imagens.

**Risco:** segmentacao ruim.

Mitigacao: comparar dois metodos e documentar falhas; se necessario, usar isolamento por bounding box como alternativa justificavel.

**Risco:** modelo com acuracia alta, mas recall ruim para defeitos.

Mitigacao: escolher modelo com base em F1 e recall da classe `rotten`, nao apenas acuracia.

## 18. Decisao final resumida

O caminho mais seguro e tecnicamente coerente e desenvolver o **cenario A**, com classificacao binaria **frutas frescas vs frutas podres**, usando pipeline classico com features de **forma, momentos de Hu, cor e textura**. O modelo principal recomendado e **Random Forest**, comparado com **SVM** e usando **Regressao Logistica** como baseline.

Essa estrategia maximiza a chance de cumprir bem a rubrica porque entrega segmentacao comparada, features justificaveis, analise de features, classificadores classicos, metricas completas, interpretabilidade basica e analise critica de erros.

## 19. Graficos e interface web

**O enunciado pede graficos e visualizacoes de avaliacao.**

Breve explicacao: a analise deve mostrar distribuicoes de features por classe, matriz de confusao, curva ROC ou matriz normalizada, comparacao de metricas entre modelos e exemplos visuais de erros.

**Interface web nao e obrigatoria, mas aparece como bonus.**

Breve explicacao: o PDF cita uma interface simples em Streamlit ou Gradio como item adicional caso o restante esteja correto.

**Decisao:** implementar um dashboard em **Streamlit**.

Breve explicacao: Streamlit e citado explicitamente no enunciado, roda bem em Python, integra tabelas, imagens e graficos rapidamente, e deixa a apresentacao final mais clara para demonstracao didatica.

**Porta alvo:** 11000, com fallback automatico.

Breve explicacao: a porta 11000 foi testada, mas o Windows recusou o bind no ambiente local; por isso o script tenta 11000 primeiro e usa 11001 se ela nao estiver disponivel.
