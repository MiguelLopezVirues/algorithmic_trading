import os
from typing import List, Optional
import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed

from .utils import get_region
from pathlib import Path

from .file_handling import FileHandler


# TO-DO:
# [] add error-handling

class TickersFetcher(FileHandler):
    # def save_dataframe(self, dataframe: pd.DataFrame, save_path: str) -> None:
    #     """
    #     Saves a DataFrame to a CSV file, creating necessary directories.
        
    #     Args:
    #         df (pd.DataFrame): The DataFrame to save.
    #         save_path (str): Path to save the CSV file.
    #     """
    #     save_path_dir = Path(save_path).parent
    #     save_path_dir.mkdir(parents=True, exist_ok=True)

    #     dataframe.to_csv(save_path)

    def fetch_and_save_historical_prices(self,
                                         symbol: str,
                                        interval: str = "1d",
                                        save_path: Optional[str] = None
                                        ) -> pd.DataFrame:
        """
        Downloads historical price data for a given symbol from Yahoo Finance and saves it as a CSV file.

        Parameters
        ----------
        symbol : str
            The ticker symbol for which to download data.
        interval : str, optional
            The time interval between data points (default is "1d").
        save_path : str, optional
            The directory path where the CSV file will be saved (default is "../data/extracted").

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the downloaded historical prices with an added "symbol" column.
        """
        historical_prices = yf.download(
            tickers=symbol,
            period="max",
            interval=interval,
            multi_level_index=False,
            progress=False
        )


        historical_prices.columns = [col.lower() for col in historical_prices.columns]
        historical_prices["symbol"] = symbol


        ticker = yf.Ticker(symbol)
        if ticker.info["quoteType"] == "EQUITY":
            region = get_region(ticker.info["country"])
            historical_prices["country"] = ticker.info["country"].lower().replace(" ","_")
            historical_prices["region"] = region
            historical_prices["industry"] = ticker.info["industryKey"]
            historical_prices["sector"] = ticker.info["sectorKey"]
            save_file = f"OHLCV/{region}/{symbol}.csv"
        else: # INDEX
            save_file = f"macro/{symbol}.csv"

        historical_prices["currency"] = ticker.info["currency"]


        # Save dataframe as csv
        if not save_path:
            save_path = f"../data/extracted/{save_file}"
        else:
            save_path = str(save_path) + f"/{symbol}.csv"
        
        self.save_dataframe_csv_file(historical_prices, save_path)

        return historical_prices

    def fetch_and_save_historical_prices_parallel(self,
                                                  symbols_list: List[str],
                                                interval: str = "1d",
                                                save_path: str = None
                                                ) -> None:
        """
        Downloads historical prices in parallel for a list of symbols and saves each as a CSV.

        Parameters
        ----------
        symbols_list : List[str]
            A list of ticker symbols to download.
        interval : str, optional
            The time interval between data points (default is "1d").
        """
        Parallel(n_jobs=-1, backend='loky')(
            delayed(self.fetch_and_save_historical_prices)(symbol, interval, save_path)
            for symbol in symbols_list
        )
