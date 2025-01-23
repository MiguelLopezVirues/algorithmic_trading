import uvicorn
import joblib
from fastapi import FastAPI
import numpy as np
import pandas as pd

threshold = 0.5 # TO PUT INSIDE A CONFIG OR SIMILAR

model = joblib.load("mock_model.joblib")
data = pd.read_parquet("mock_data.parquet")
class_names = np.array(["buy","sell"])

app = FastAPI()

@app.get("/")
def reed_root():
    return {"message":"API working"}

@app.post("/predict")
def predict(data: dict):
    # pd.DataFrame(mock_data.to_dict(orient="records"))
    data_to_predict = pd.DataFrame(data)
    
    y_probs = model.predict_proba(data_to_predict)[:,1]

    y_pred = (y_probs > threshold).astype(int)

    data_to_predict["buy"] = y_pred

    return {"predictions": y_pred.tolist()}
    
    


