import uvicorn
import joblib
from fastapi import FastAPI
import numpy as np
import pandas as pd


model = joblib.load("predict/mock_model.joblib")
data = pd.read_parquet("predict/mock_data.parquet")

app = FastAPI()

@app.get("/")