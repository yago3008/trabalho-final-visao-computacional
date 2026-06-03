# Guia de implementacao orientada

Este guia organiza o que a equipe deve implementar. Ele nao substitui o trabalho da equipe; serve como mapa tecnico para evitar perda de tempo e erros metodologicos.

## 1. Notebook 01 - segmentacao

Objetivo:

- carregar exemplos das classes `fresh` e `rotten`;
- testar dois metodos classicos de isolamento da fruta;
- salvar exemplos visuais de acertos e falhas.

Metodos sugeridos:

- limiar em HSV;
- Otsu em canal de intensidade ou saturacao, seguido de operacoes morfologicas.

O que observar:

- se a mascara pega a fruta inteira;
- se partes do fundo entram na mascara;
- se manchas escuras sao preservadas;
- se frutas claras ou amareladas confundem o limiar.

## 2. Notebook 02 - features

Objetivo:

- calcular features manuais por imagem;
- gerar `X.csv` e `y.csv`;
- garantir que as features usem a mascara da fruta.

Familias obrigatorias:

- forma: area, perimetro, circularidade, eccentricity, solidity, extent;
- inerciais: 7 momentos de Hu em escala logaritmica;
- cor: medias RGB ou HSV, preferencialmente dentro da mascara;
- textura: GLCM ou LBP.

Conferencias importantes:

- verificar se nao existem valores `NaN`;
- manter o nome da imagem junto das features para rastrear erros;
- conferir se as classes estao balanceadas.

## 3. Notebook 03 - classificacao

Objetivo:

- treinar pelo menos dois classificadores classicos;
- comparar metricas;
- escolher o melhor modelo com justificativa.

Modelos recomendados:

- Regressao Logistica como baseline;
- SVM como comparacao forte;
- Random Forest como candidato principal.

Boas praticas:

- usar divisao estratificada;
- ajustar `StandardScaler` apenas com treino;
- manter teste separado ate a avaliacao final;
- usar `random_state` fixo.

Metricas obrigatorias:

- acuracia;
- precisao;
- recall;
- F1-score;
- matriz de confusao.

Metrica principal recomendada:

- F1 da classe `rotten` ou F1 macro.

Explicacao breve: em inspecao visual, deixar passar um item defeituoso costuma ser mais grave do que rejeitar um item bom.

## 4. Analise de features

Gerar:

- boxplots por classe;
- medias ou medianas por classe;
- matriz de correlacao;
- PCA para visualizacao;
- SelectKBest;
- importancia de variaveis do Random Forest.

Perguntas que a equipe deve responder:

- quais features separam melhor `fresh` de `rotten`?
- cor foi mais importante que textura?
- forma realmente ajudou ou foi pouco discriminativa?
- houve features redundantes?

## 5. Analise de erros

Separar de 5 a 10 exemplos em que o modelo errou.

Para cada erro, investigar:

- a segmentacao falhou?
- a imagem tem iluminacao diferente?
- a fruta esta ambigua?
- o fundo interferiu?
- a imagem parece ser aumento artificial de outra?

Separar:

- falsos positivos: fruta fresca classificada como podre;
- falsos negativos: fruta podre classificada como fresca.

## 6. Criterios para decisao do melhor modelo

Escolher o modelo final considerando:

- F1-score;
- recall da classe `rotten`;
- matriz de confusao;
- estabilidade em validacao cruzada;
- interpretabilidade;
- custo computacional;
- facilidade de explicar no relatorio e no video.

Decisao esperada:

- se Random Forest vencer ou empatar, ele tende a ser a melhor escolha pela interpretabilidade e robustez;
- se SVM vencer claramente, justificar com as metricas e discutir que ele e menos interpretavel;
- se Regressao Logistica for competitiva, destacar que um modelo simples pode ser suficiente.

## 7. Pontos que eu posso revisar depois

Quando a equipe tiver implementado trechos do projeto, posso ajudar a:

- revisar funcoes de segmentacao;
- explicar erros de OpenCV, scikit-image ou scikit-learn;
- verificar vazamento de dados;
- revisar graficos e metricas;
- apontar inconsistencias entre codigo, resultados e conclusoes;
- sugerir melhorias pontuais em codigo ja escrito.
