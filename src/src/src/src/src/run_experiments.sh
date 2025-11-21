#!/usr/bin/env bash
# run_experiments.sh
# Usage: ./run_experiments.sh
set -euo pipefail

echo "1) Generate synthetic data..."
python src/generate_synthetic.py

echo "2) Preprocess data..."
python src/data_preprocessing.py data/synthetic_data.csv

echo "3) Train models..."
python src/modeling.py

echo "4) Run SHAP analyses..."
python src/shap_analysis.py

echo "All done. Check reports/shap/ and models/ directories."
