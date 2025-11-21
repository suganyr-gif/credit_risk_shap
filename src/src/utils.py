# src/utils.py
import joblib
import os

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_pickle(path):
    return joblib.load(path)
