from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("model.joblib")

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])[0]
    return {"prediction": int(prediction)}
