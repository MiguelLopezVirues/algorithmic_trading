import time

from support.data_extraction import TickersFetcher
from support.data_transformation import TickerExtender, TechnicalIndicators
from support.data_load import MongoDBHandler

from support.utils import load_tickers_to_include

from dotenv import load_dotenv
import os

load_dotenv()
db_user = os.environ.get("MONGO_DB_USER")
db_pass = os.environ.get("MONGO_DB_PASS")
db_host = os.environ.get("MONGO_HOST")
database = os.environ.get("MONGO_DB_NAME")
mongodb_options = os.environ.get("MONGO_OPTIONS")


def tickers_etl():
    start_time = time.time()
    # instantiate objects for extraction, transformation and load
    tickers_fetcher = TickersFetcher()
    ticker_extender = TickerExtender()
    db_handler = MongoDBHandler(db_user=db_user, db_pass=db_pass, host=db_host, options=mongodb_options)
    
    # get list of tickers to process
    symbols_list = load_tickers_to_include()["TICKERS"]

    # 1.Extract
    print("Extracting...\n")
    tickers_fetcher.fetch_and_save_historical_prices_parallel(symbols_list)

    # 2.Transform
    print("Transforming...")
    print("Enriching ticker dataframes.")
    ticker_df_list = ticker_extender.transform_daily_tickers_parallel("../data/extracted/OHLCV")
    print("Merging into one collection.")
    merged_ticker_df = ticker_extender.merge_tickers(ticker_df_list)

    # 3.Load
    print("Loading to database...\n")
    # db_handler.connect_to_database(database)
    # collection = db_handler.create_collection("stocks_tickers")
    # documents = merged_ticker_df.to_dict(orient='records')
    # db_handler.insert_documents(collection_name=collection.name, documents=documents)
    end_time = time.time()

    print(f"Tickers ETL took {end_time - start_time:.2f} seconds")


def main():
    tickers_etl()

   
if __name__ == "__main__":
    main()
