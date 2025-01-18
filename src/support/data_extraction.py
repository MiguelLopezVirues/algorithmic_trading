import os
from typing import List, Optional
import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed


# TO-DO:
# [] add error-handling

class TickersFetcher():
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

        save_path = save_path or "../data/extracted"
        os.makedirs(name=save_path, exist_ok=True)

        csv_path = os.path.join(save_path, f"{symbol}.csv")
        historical_prices.to_csv(csv_path)

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
