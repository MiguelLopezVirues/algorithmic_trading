import os
from typing import List, Optional, Dict, Tuple
import polars as pl  
import yfinance as yf
from joblib import Parallel, delayed
import pandas_datareader as pdr
import eurostat

from .utils import get_region, get_index_country
from pathlib import Path

from .file_handling import FileHandler

base_dir = os.path.dirname(__file__)

class YahooFinanceFetcher(FileHandler):
    def fetch_historical_prices(self,
                                symbol: str,
                                interval: str = "1d"
                                ) -> pl.DataFrame:
        
        # Download historical data and convert to polars DataFrame
        historical_prices_df = pl.from_pandas(  # Convert to polars
            yf.download( 
                tickers=symbol,
                period="max",
                interval=interval,
                multi_level_index=False,
                progress=False
            ), 
            include_index=True
        )

        # Standardize column names and add symbol
        historical_prices_df = (
            historical_prices_df
            .rename({col: col.lower() for col in historical_prices_df.columns})
            .with_columns(pl.lit(symbol).alias("symbol"))
        )

        return historical_prices_df

    def fetch_ticker_info(self,
                        symbol: str,
                        historical_prices_df: pl.DataFrame
                        ) -> Tuple[pl.DataFrame, Dict]: 
        
        # get fundamentals information
        ticker = yf.Ticker(symbol)
        currency = ticker.info.get("currency", "unknown")
        historical_prices_df = historical_prices_df.with_columns(
            pl.lit(currency).alias("currency") # pl.lit() prevents Polars from thinking the currency value is a column name
        )

        # if EQUITY (stock) add stock specific info
        quote_type = ticker.info.get("quoteType")

        if quote_type == "EQUITY":
            country = ticker.info.get("country", "").lower().replace(" ", "_")
            region = get_region(ticker.info["country"])
            
            historical_prices_df = historical_prices_df.with_columns([
                pl.lit(ticker.info.get("industryKey", "unknown")).alias("industry"),
                pl.lit(ticker.info.get("sectorKey", "unknown")).alias("sector")
            ])
        else:
            
            country = get_index_country(symbol)
            region = get_region(country)

        historical_prices_df = historical_prices_df.with_columns([
                pl.lit(country).alias("country"),
                pl.lit(region).alias("region"),
        ])


        # besides df, return quote_type and region for save_path modification outside function
        return historical_prices_df, quote_type, region

    def fetch_save_prices_info(
        self,
        symbol: str,
        interval: str = "1d",
        save_path: Optional[str] = None
    ) -> pl.DataFrame:  
        """
        Downloads historical price data for a given symbol from Yahoo Finance and saves it as a parquet file.

        Parameters
        ----------
        symbol : str
            The ticker symbol for which to download data.
        interval : str, optional
            The time interval between data points (default is "1d").
        save_path : str, optional
            The directory path where the parquet file will be saved (default is "../data/extracted").

        Returns
        -------
        pl.DataFrame
            A polars DataFrame containing the downloaded historical prices with added columns.
        """
        # Download historical data and convert to polars DataFrame
        historical_prices_df = self.fetch_historical_prices(symbol=symbol,
                                                         interval=interval)
        
        # Add non-historical fundamental information
        historical_prices_df, quote_type, region = self.fetch_ticker_info(symbol=symbol, 
                                                                        historical_prices_df=historical_prices_df)
        

        # Handle path construction and saving
        if not save_path:
            # Different path for stocks (EQUITY) or macroindicators
            if quote_type == "EQUITY": # stocks
                save_file = f"OHLCV/{region}/{symbol}.parquet"

            else:  # macroindicators
                save_file = f"macro/{symbol}.parquet"

            save_path = Path(base_dir) / "../../data/extracted" / save_file

        else:
            save_path = Path(save_path) / f"{symbol}.parquet"

        # Create directories and save
        self.save_dataframe_parquet_file(historical_prices_df, save_path)

        return historical_prices_df

    def fetch_save_prices_info_parallel(
        self,
        symbols_list: List[str],
        interval: str = "1d",
        save_path: str = None
    ) -> None:
        """
        Downloads historical prices in parallel for a list of symbols and saves each as a parquet.
        """
        Parallel(n_jobs=-1, backend='loky')(
            delayed(self.fetch_save_prices_info)(symbol, interval, save_path)
            for symbol in symbols_list
        )
    


class PDRFetcher(FileHandler):
    def fetch_save_historical_data(self, symbol: str, symbol_lag: Tuple[int, str],  start: str = "1955-01-01", save_path: str = None):
        pdr_data = pl.from_pandas(pdr.DataReader(symbol, "fred", start=start), include_index=True)
        
        pdr_data = pdr_data.rename({col: col.lower() for col in pdr_data.columns})

        if symbol_lag[0]:
            lag, period = symbol_lag
            pdr_data = pdr_data.select("date",
                                        pl.exclude("date").name.suffix(f"_prev{lag}_{period}"))\
                                .with_columns(pl.col("date").dt.offset_by(f"{lag}{period}"))

        save_path = save_path or Path(base_dir) / f"../../data/extracted/macro/{symbol}.parquet"

        self.save_dataframe_parquet_file(pdr_data, save_path)

        return pdr_data
    
    def fetch_save_prices_info_parallel(
                                        self,
                                        symbols_dict
                                    ) -> None:
        """
        Downloads historical prices in parallel for a list of symbols and saves each as a parquet.
        """
        Parallel(n_jobs=-1, backend='loky')(
            delayed(self.fetch_save_historical_data)(symbol, symbol_lag)
            for symbol, symbol_lag in symbols_dict.items()
        )


def fetch_euro_yield(save_path: str = None):
    filter_pars = {"yld_curv": ["SPOT_RT"],
    "bonds": ["CGB_EA_AAA"],
    "maturity": ["Y1","Y5","Y10"]}

    code = 'irt_euryld_d'
    eurostat_euro_yield_df = pl.from_pandas(eurostat.get_data_df(code, flags=False, filter_pars=filter_pars,  verbose=True))

    save_path = save_path or Path(base_dir) / f"../../data/extracted/macro/eurostat_euro_yield.parquet"

    eurostat_euro_yield_df.write_parquet(save_path)

    return eurostat_euro_yield_df