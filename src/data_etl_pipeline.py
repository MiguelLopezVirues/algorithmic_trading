import time
import polars as pl

from support.data_extraction import YahooFinanceFetcher, PDRFetcher, fetch_euro_yield
from support.data_transformation import TickerExtender, TechnicalIndicators
from support.data_load import MongoDBHandler

from support.utils import load_config
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()
db_user = os.environ.get("MONGO_DB_USER")
db_pass = os.environ.get("MONGO_DB_PASS")
db_host = os.environ.get("MONGO_HOST")
database = os.environ.get("MONGO_DB_NAME")
mongodb_options = os.environ.get("MONGO_OPTIONS")

# instantiate objects for extraction, transformation and load
yahoo_fetcher = YahooFinanceFetcher()
pdr_fetcher = PDRFetcher()
ticker_extender = TickerExtender()
db_handler = MongoDBHandler(db_user=db_user, db_pass=db_pass, host=db_host, options=mongodb_options)

def extract_tickers():
    base_dir = os.path.dirname(__file__)
    # get list of tickers to process
    tickers = load_config("tickers_to_include.yaml")
    symbols_list = tickers["STOCKS"]
    macro_list = tickers["MACRO"]
    macro_pdr_dict = tickers["MACRO_PDR"]       

    # 1.Extract
    # yahoo finance
    print("Extracting stocks...\n")
    yahoo_fetcher.fetch_save_prices_info_parallel(symbols_list)
    print("Extracting macroindicators from yf...\n")
    yahoo_fetcher.fetch_save_prices_info_parallel(macro_list, save_path = Path(base_dir) / "../data/extracted/macro")

    # pdr finance
    print("Extracting macroindicators from pdr...\n")
    pdr_fetcher.fetch_save_prices_info_parallel(macro_pdr_dict, dir_path = Path(base_dir) / "../data/extracted/macro")

    # euroyield
    fetch_euro_yield(save_path = Path(base_dir) / "../data/extracted/macro/eurostat_euro_yield.parquet")



def tickers_etl():
    base_dir = os.path.dirname(__file__)
    start_time = time.time()

    # 1.Extract
    extract_tickers()


    # 2.Transform
    print("Transforming...")
    print("Enriching ticker dataframes.")
    ticker_df_list = ticker_extender.transform_daily_tickers_parallel(Path(base_dir) / "../data/extracted/OHLCV")
    print("Merging into one collection.")
    merged_ticker_df = ticker_extender.merge_tickers(ticker_df_list)

    base_dir = os.path.dirname(__file__)
    merged_ticker_df.write_parquet(Path(base_dir) / "../data/transformed/dataset.parquet")


    # 3.Load
    print("Loading to database...\n")
    # db_handler.connect_to_database(database)
    # collection = db_handler.check_create_collection("stocks_tickers")
    # documents = merged_ticker_df.to_dict(orient='records')
    # db_handler.insert_documents(collection_name=collection.name, documents=documents)
    # end_time = time.time()

    print(f"Tickers ETL took {end_time - start_time:.2f} seconds")


   
if __name__ == "__main__":
    tickers_etl()
