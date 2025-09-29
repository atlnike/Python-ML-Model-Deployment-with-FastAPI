from fastapi import FastAPI
from app.model import load_model
from app.predict import IrisFeatures, predict_class

app = FastAPI(title="Iris Classifier API", version="1.0")

model = load_model()

@app.get("/")
def read_root():
    return {"message": "Iris Classifier API is up and running!"}

@app.post("/predict")
def predict(iris: IrisFeatures):
    prediction = predict_class(model, iris)
    return {"predicted_class": prediction}
