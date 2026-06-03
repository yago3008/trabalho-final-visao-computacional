$ErrorActionPreference = "Stop"

$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    & "C:\Users\yago.martins.SOOW\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
    & $Python -m pip install -r requirements.txt
}

& $Python scripts\prepare_dataset.py --train-per-class 100 --test-per-class 40
& $Python scripts\run_segmentation_examples.py --per-class 5
& $Python scripts\run_features.py --segmentation hsv --size 256
& $Python scripts\analyze_features.py --k 12
& $Python scripts\train_models.py
& $Python scripts\copy_error_examples.py --errors outputs\modelos\random_forest_test_errors.csv --out outputs\erros\random_forest --limit 10
& $Python scripts\explain_model.py --model outputs\modelos\random_forest.joblib --split test --top 20

