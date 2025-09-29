import requests

url = "http://localhost:8000/predict"
sample = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

res = requests.post(url, json=sample)
print(res.json())
