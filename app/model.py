import joblib
import os

MODEL_PATH = os.path.join("model", "iris_clf.pkl")

def load_model():
    return joblib.load(MODEL_PATH)
