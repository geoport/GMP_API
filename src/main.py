from fastapi import FastAPI
from model import Prediction
from predict import predict_gmpe
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import keras

print(keras.__version__, "-------------------------------")


@app.post("/predict")
async def predict(prediction_data: Prediction):
    output = predict_gmpe(prediction_data)

    return output
