# src/shap_analysis.py
"""
Run SHAP analysis for a model. Produces summary and dependence plots (saved as PNG).
Requires matplotlib.
"""
import joblib
import shap
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

PROCESSED_PKL = "data/processed.pkl"
MODELS_DIR = "models"
OUT_DIR = "reports/shap"

def run_shap(model_path, processed_pkl=PROCESSED_PKL, out_dir=OUT_DIR, sample_size=1000):
    os.makedirs(out_dir, exist_ok=True)
    model = joblib.load(model_path)
    data = joblib.load(processed_pkl)
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Evaluate AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"AUC on test set: {auc:.4f}")
    else:
        print("Model has no predict_proba; skipping AUC")

    # background / sample
    background = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    # choose explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(background)
    except Exception as e:
        print("TreeExplainer failed, trying Kernel/Linear Explainer:", e)
        try:
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(background)
        except Exception as e2:
            print("Fallback KernelExplainer (slow):", e2)
            explainer = shap.KernelExplainer(model.predict_proba, background.iloc[:100])
            shap_values = explainer.shap_values(background.iloc[:200])

    # For TreeExplainer on binary classification, shap_values may be list
    if isinstance(shap_values, list):
        # shap_values[1] corresponds to positive class usually
        sv = shap_values[1]
    else:
        sv = shap_values

    # summary plot
    plt.figure(figsize=(8,6))
    shap.summary_plot(sv, background, show=False)
    plt.tight_layout()
    summary_path = os.path.join(out_dir, os.path.basename(model_path).replace(".pkl","") + "_summary.png")
    plt.savefig(summary_path)
    plt.close()
    print("Saved summary plot to", summary_path)

    # dependence plot for top feature
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argmax(mean_abs)
    top_feat = background.columns[top_idx]
    print("Top feature by mean(|SHAP|):", top_feat)
    plt.figure(figsize=(8,6))
    shap.dependence_plot(top_feat, sv, background, show=False)
    dep_path = os.path.join(out_dir, os.path.basename(model_path).replace(".pkl","") + f"dependence{top_feat}.png")
    plt.tight_layout()
    plt.savefig(dep_path)
    plt.close()
    print("Saved dependence plot to", dep_path)

if _name_ == "_main_":
    # Default: run on both saved models
    for m in ["models/xgboost.pkl", "models/logistic.pkl"]:
        if os.path.exists(m):
            run_shap(m)
        else:
            print("Model not found:", m)
