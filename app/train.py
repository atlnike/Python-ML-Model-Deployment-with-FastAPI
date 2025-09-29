import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    joblib.dump(clf, "model/iris_clf.pkl")
    print("✅ Model trained and saved!")

if __name__ == "__main__":
    train_model()
