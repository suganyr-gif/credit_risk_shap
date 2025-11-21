# src/data_preprocessing.py
"""
Simple preprocessing pipeline:
- Reads CSV
- Basic imputing
- Encodes small categoricals with one-hot
- Creates a train/test split and saves processed objects
Outputs: data/processed.pkl
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

OUTPUT_PKL = "data/processed.pkl"

def build_and_save(input_csv, output_pkl=OUTPUT_PKL, random_state=42):
    print(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv)
    # drop safe-guard columns if present
    for col in ["name","ssn","id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    assert "default" in df.columns, "Input CSV must contain 'default' target column."
    y = df["default"].astype(int)
    X = df.drop(columns=["default"])
    # detect column types
    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

    # simple imputer for numeric and categorical
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder="drop")

    # Fit preprocessor on full dataset (we'll save and reuse)
    print("Fitting preprocessors ...")
    X_processed = preprocessor.fit_transform(X)

    # Build feature names
    num_features = numeric_cols
    # get cat feature names from onehot
    cat_feature_names = []
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(cat_cols)
        cat_feature_names = cat_names.tolist()

    feature_names = num_features + cat_feature_names
    X_proc_df = pd.DataFrame(X_processed, columns=feature_names)
    print("Splitting train/test ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc_df, y, test_size=0.2, stratify=y, random_state=random_state
    )
    print(f"Saving processed objects to {output_pkl} ...")
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    joblib.dump({
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }, output_pkl)
    print("Done.")

if _name_ == "_main_":
    import sys
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_data.csv"
    build_and_save(input_csv)
