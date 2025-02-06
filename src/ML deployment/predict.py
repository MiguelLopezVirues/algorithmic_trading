import polars as pl
import joblib
from skforecast.utils import load_forecaster
from support.data_transformation import TickerExtender
from support.data_load import MongoDBHandler

import os
from pathlib import Path
base_dir = os.path.dirname(__file__)

from datetime import datetime

from typing import List

import boto3


def get_secret_from_ssm(secret_name):
    ssm = boto3.client('ssm', region_name='eu-west-1') 
    try:
        response = ssm.get_parameter(
            Name=secret_name,
            WithDecryption=True  # Decrypt the AWS Parameter SecureString
        )
        return response['Parameter']['Value']
    
    except Exception as e:
        print(f"Error retrieving secret: {str(e)}")
        return None


def load_aws_secrets(secret_names_list):
    return {secret_name: get_secret_from_ssm(secret_name) for secret_name in secret_names_list}


# Retrieve MongoDB credentials and declare db handler
secrets = load_aws_secrets(["MONGO_DB_USER","MONGO_DB_PASS","MONGO_HOST","MONGO_DB_NAME","MONGO_OPTIONS"])
db_handler = MongoDBHandler(
    db_user=secrets["MONGO_DB_USER"], 
    db_pass=secrets["MONGO_DB_PASS"], 
    host=secrets["MONGO_HOST"], 
    options=secrets["MONGO_OPTIONS"]
)

def predict(model, exog_dict_prediction, last_window=None, steps: int = 5, interval: List[int] = [25, 75], n_boot = 2000):
    if last_window is None:
        predictions = model.predict_interval(steps=steps, use_in_sample_residuals=False, exog=exog_dict_prediction, interval = interval, n_boot = n_boot)
    else:
        predictions = model.predict_interval(steps=steps, use_in_sample_residuals=False, exog=exog_dict_prediction, interval = interval, n_boot = n_boot, last_window=last_window)  # Replace with the correct method
    
    return predictions

def load_predictions_db(predictions_df, db_handler):
    db_handler.connect_to_database(secrets["MONGO_DB_NAME"])
    collection = db_handler.check_create_collection("stocks_predictions")
    documents = predictions_df.to_dict(orient='records')
    if db_handler.insert_documents(collection_name=collection.name, documents=documents):
        print("Prediction database upload succesful.")


if __name__ == "__main__":
    # Load model
    # model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'model.pkl')
    model = load_forecaster(Path(base_dir) / "model/forecaster_recursive_multiseries.joblib")

    # Check if last_window data is provided
    last_window = None
    # if os.environ.get('USE_LAST_WINDOW') == 'True':
    #     last_window_path = os.path.join(os.environ['SM_INPUT_DIR'], 'last_window.parquet')
    #     last_window = pl.read_parquet(last_window_path)

    # Generate exog
    ticker_extender = TickerExtender()
    exog_dict = ticker_extender.generate_exog_dict(Path(base_dir) / "data/transformed/OHLCV/")

    # Generate predictions
    predictions = predict(model, exog_dict, last_window=None, steps = 5)

    # Load predictions to MongoDB Database
    predictions["target_date"] = predictions.index
    predictions["forcast_creation"] = datetime.today()

    print("Loading predictions to database...\n")
    load_predictions_db(predictions, db_handler=db_handler)
        

