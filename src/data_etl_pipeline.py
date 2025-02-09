import time
import polars as pl
import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from src.support.data_extraction import YahooFinanceFetcher, PDRFetcher, fetch_euro_yield
from src.support.data_transformation import TickerExtender
from src.support.data_load import MongoDBHandler
from src.support.utils import load_config

# Load environment variables
load_dotenv()
db_user = os.environ.get("MONGO_DB_USER")
db_pass = os.environ.get("MONGO_DB_PASS")
db_host = os.environ.get("MONGO_HOST")
database = os.environ.get("MONGO_DB_NAME")
mongodb_options = os.environ.get("MONGO_OPTIONS")

# Instantiate objects for extraction, transformation, and load
yahoo_fetcher = YahooFinanceFetcher()
pdr_fetcher = PDRFetcher()
ticker_extender = TickerExtender()
db_handler = MongoDBHandler(db_user=db_user, db_pass=db_pass, host=db_host, options=mongodb_options)


def extract_tickers() -> None:
    """
    Extracts financial data for stocks and macroeconomic indicators from various sources.

    Params:
    ---------
    None

    Returns:
    ------
    None
    """
    base_dir = os.path.dirname(__file__)

    # Get list of tickers to process
    tickers: Dict[str, Any] = load_config("tickers_to_include.yaml")
    symbols_list = tickers["STOCKS"]
    macro_list = tickers["MACRO"]
    macro_pdr_dict = tickers["MACRO_PDR"]       

    # Extraction process
    print("Extracting stocks...\n")
    yahoo_fetcher.fetch_save_prices_info_parallel(symbols_list)

    print("Extracting macroindicators from yf...\n")
    yahoo_fetcher.fetch_save_prices_info_parallel(macro_list, save_path=Path(base_dir) / "../data/extracted/macro")

    print("Extracting macroindicators from pdr...\n")
    pdr_fetcher.fetch_save_prices_info_parallel(macro_pdr_dict, dir_path=Path(base_dir) / "../data/extracted/macro")

    print("Extracting Euro Yield data...\n")
    fetch_euro_yield(save_path=Path(base_dir) / "../data/extracted/macro/eurostat_euro_yield.parquet")


def transform_tickers(forecast_horizon: int = 5) -> pl.DataFrame:
    """
    Transforms extracted ticker data by enriching and merging into a single dataset.

    Params:
    ---------
    forecast_horizon : int, optional
        The forecast horizon in days (default is 5).

    Returns:
    ------
    pl.DataFrame
        The transformed and merged dataset containing enriched ticker data.
    """
    base_dir = os.path.dirname(__file__)

    print("Transforming...")
    print("Enriching ticker dataframes.")
    ticker_df_list = ticker_extender.transform_daily_tickers_parallel_experiment(
        Path(base_dir) / "../data/extracted/OHLCV", 
        forecast_horizon=forecast_horizon
    )

    print("Merging into one collection.")
    merged_ticker_df = ticker_extender.merge_tickers(ticker_df_list)

    merged_ticker_df.write_parquet(Path(base_dir) / "../data/transformed/dataset.parquet")

    return merged_ticker_df



def tickers_etl() -> None:
    """
    Executes the full ETL (Extract, Transform, Load) pipeline for financial tickers.

    Params:
    ---------
    None

    Returns:
    ------
    None
    """
    start_time = time.time()

    # 1.Extract
    extract_tickers()


    # 2.Transform
    documents_df = transform_tickers()

    # 3.Load
    print("Loading to database...\n")
    # COMMENTED ON PURPOSE: 
    # MogoDB Atlas cannot take all the data. Besides, loading to database is not immediately necessary, as extraction and transformation are very fast.
    # documents = documents_df.to_dict(orient='records')  # Convert DataFrame rows into a list of dictionaries
    # db_handler.load_to_mongodb(documents=documents, database=database, collection_name="transformed_data")

    end_time = time.time()

    print(f"Tickers ETL took {end_time - start_time:.2f} seconds")

   
if __name__ == "__main__":
    tickers_etl()
