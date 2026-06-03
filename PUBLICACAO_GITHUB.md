# Guia de publicacao no GitHub

Este arquivo deixa o projeto pronto para a etapa externa de publicacao. A criacao do repositorio remoto, login no GitHub e postagem no Blackboard precisam ser feitos manualmente.

## 1. Conferir arquivos principais

Antes de publicar, confira se estes arquivos estao presentes:

- `README.md`
- `requirements.txt`
- `X.csv`
- `y.csv`
- `notebooks/`
- `scripts/`
- `src/`
- `outputs/figuras/`
- `outputs/modelos/metricas_modelos.csv`
- `outputs/modelos/matrizes_confusao/`
- `outputs/modelos/roc/`
- `outputs/erros/random_forest/`
- `outputs/xai/`
- `outputs/relatorio_tecnico.pdf`
- `ANALISE_ERROS.md`
- `CHECKLIST_REQUISITOS.md`

## 2. Arquivos grandes ignorados

O `.gitignore` foi pensado para nao enviar:

- ambiente virtual `.venv/`;
- cache local `.uv-cache/`;
- zip bruto do dataset;
- dataset expandido bruto;
- imagens processadas usadas apenas como cache;
- modelos `.joblib`;
- caches `__pycache__/`;
- previews temporarios do PDF.

O repositorio continua reprodutivel porque o README explica como preparar o dataset e executar o pipeline.

Observacao: a aba **Imagens** do dashboard usa os caminhos de `data/processed/`. Como essa pasta e gerada localmente e fica ignorada no Git, quem clonar o repositorio deve rodar o preparo do dataset antes de usar a galeria completa.

## 3. Comandos locais

Se ainda nao houver repositorio Git inicializado:

```powershell
git init
git add .
git commit -m "Implementa pipeline de inspecao visual"
```

Depois de criar um repositorio publico no GitHub, conectar o remoto:

```powershell
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
git push -u origin main
```

Se o remoto ja existir:

```powershell
git remote set-url origin https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
git push -u origin main
```

## 4. Validacao antes do push

Execute:

```powershell
& .\.venv\Scripts\python.exe -m py_compile .\dashboard_app.py
& .\.venv\Scripts\python.exe .\scripts\build_report.py
powershell.exe -ExecutionPolicy Bypass -File .\run_demo_pipeline.ps1
```

O ultimo comando reexecuta a rodada demonstrativa e pode sobrescrever outputs com os mesmos parametros de exemplo.

## 5. Blackboard

Postar no Blackboard:

- link do repositorio GitHub publico;
- link do video no YouTube como unlisted.

Prazo informado no enunciado: `07/06/2026`.
