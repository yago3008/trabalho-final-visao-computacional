# Dataset

Dataset baixado:

- Arquivo: `data/raw/fruits_fresh_rotten.zip`
- Tamanho aproximado: 3,8 GB
- Origem: Kaggle, `sriramr/fruits-fresh-and-rotten-for-classification`
- Total listado no ZIP: 27198 entradas

## Estrutura observada no ZIP

O arquivo contem imagens separadas em `train` e `test`, com pastas de classes:

- `freshapples`
- `freshbanana`
- `freshoranges`
- `rottenapples`
- `rottenbanana`
- `rottenoranges`

## Decisao recomendada para o projeto

Usar classificacao binaria:

- `fresh`: imagens vindas de `freshapples`, `freshbanana`, `freshoranges`
- `rotten`: imagens vindas de `rottenapples`, `rottenbanana`, `rottenoranges`

Explicacao breve: essa decisao fica alinhada ao cenario A do enunciado e reduz a complexidade do problema. Em vez de treinar um modelo para distinguir seis classes, a equipe avalia se o item esta dentro do padrao ou fora do padrao.

## Cuidado metodologico

Evitem misturar imagens aumentadas artificialmente entre treino e teste se elas vierem da mesma imagem original. Como o dataset contem imagens com nomes como `rotated_by_...`, pode haver versoes transformadas da mesma fruta. Isso pode inflar artificialmente as metricas.

Uma estrategia mais segura e:

1. comecar usando a divisao original `train` e `test` do dataset;
2. criar uma divisao de validacao apenas a partir do treino;
3. manter o teste separado ate a avaliacao final;
4. discutir no relatorio que o dataset possui imagens aumentadas.

## Comandos uteis

Listar as primeiras entradas:

```powershell
tar -tf .\data\raw\fruits_fresh_rotten.zip | Select-Object -First 40
```

Contar entradas:

```powershell
tar -tf .\data\raw\fruits_fresh_rotten.zip | Measure-Object -Line
```

Extrair quando a equipe for trabalhar localmente:

```powershell
tar -xf .\data\raw\fruits_fresh_rotten.zip -C .\data\raw
```

Observacao: a extracao pode ocupar varios GB adicionais.
