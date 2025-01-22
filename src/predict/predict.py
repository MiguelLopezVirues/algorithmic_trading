import uvicorn
import joblib
from fastapi import FastAPI
import numpy as np
import pandas as pd

threshold = 0.6 # TO PUT INSIDE A CONFIG OR SIMILAR

model = joblib.load("predict/mock_model.joblib")
data = pd.read_parquet("predict/mock_data.parquet")
class_names = np.array(["buy","sell"])

app = FastAPI()

@app.get("/")
def reed_root():
    return {"message":"API working"}

@app.get("/predict")
def predict(data: dict):
    # pd.DataFrame(mock_data.to_dict(orient="records"))
    data_to_predict = pd.DataFrame(data).values
    
    y_probs = model.predict_proba(data_to_predict)[:,1]

    y_pred = 

