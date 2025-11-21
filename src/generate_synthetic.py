# src/generate_synthetic.py
"""
Generate a synthetic credit-risk dataset for prototyping.
Outputs: data/synthetic_data.csv
"""
import numpy as np
import pandas as pd
import os

def generate_synthetic(n=5000, seed=42, out_path="data/synthetic_data.csv"):
    np.random.seed(seed)
    age = np.random.randint(18, 75, n)
    income = np.random.exponential(50000, n) + 10000
    credit_score = np.clip(np.random.normal(650, 70, n), 300, 850)
    loan_amount = np.random.exponential(15000, n)
    employment_years = np.random.poisson(3, n)
    home_ownership = np.random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n, p=[0.5,0.2,0.25,0.05])
    purpose = np.random.choice(["debt_consolidation","credit_card","home_improvement","major_purchase","other"], n)
    marital_status = np.random.choice(["single","married","divorced","widowed"], n, p=[0.45,0.45,0.07,0.03])
    debt_to_income = loan_amount / (income + 1e-6)
    # synthetic logit to create default probability
    logit = -5 + 0.02*(70 - age) + 0.005*(loan_amount/1000) - 0.01*(credit_score - 650) + 3*(debt_to_income > 0.4)
    prob = 1 / (1 + np.exp(-logit))
    default = (np.random.rand(n) < prob).astype(int)
    df = pd.DataFrame({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "marital_status": marital_status,
        "debt_to_income": debt_to_income,
        "default": default
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic data to {out_path}")

if _name_ == "_main_":
    generate_synthetic()
