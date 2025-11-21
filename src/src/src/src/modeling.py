# src/modeling.py
"""
Trains two models: LogisticRegression (simple) and XGBoost (complex).
Saves best-fitted models as joblib files under models/
"""
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
import numpy as np

PROCESSED_PKL = "data/processed.pkl"
OUT_DIR = "models"

def train_all(processed_pkl=PROCESSED_PKL, out_dir=OUT_DIR):
    data = joblib.load(processed_pkl)
    X_train = data["X_train"]
    y_train = data["y_train"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression (with default scaler already applied)
    print("Training Logistic Regression...")
    log_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    log_params = {"C": [0.01, 0.1, 1, 10]}
    log_gs = GridSearchCV(log_clf, param_grid=log_params, cv=cv, scoring="roc_auc", n_jobs=-1)
    log_gs.fit(X_train, y_train)
    print("Best Logistic params:", log_gs.best_params_)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(log_gs.best_estimator_, os.path.join(out_dir, "logistic.pkl"))

    # XGBoost
    print("Training XGBoost...")
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    xgb_params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1]
    }
    xgb_gs = GridSearchCV(xgb_clf, param_grid=xgb_params, cv=cv, scoring="roc_auc", n_jobs=-1)
    xgb_gs.fit(X_train, y_train)
    print("Best XGBoost params:", xgb_gs.best_params_)
    joblib.dump(xgb_gs.best_estimator_, os.path.join(out_dir, "xgboost.pkl"))

    print(f"Models saved to {out_dir}")

if _name_ == "_main_":
    train_all()
