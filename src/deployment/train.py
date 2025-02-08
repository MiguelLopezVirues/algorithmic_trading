import time
import polars as pl
import pandas as pd

from support.data_extraction import YahooFinanceFetcher, PDRFetcher, fetch_euro_yield
from support.data_transformation import TickerExtender
from support.model_evaluation import long_series_exog_to_dict, evaluate_recursive_multiseries_2

from sklearn.ensemble import HistGradientBoostingRegressor

from support.utils import load_config

from datetime import datetime

import os
from pathlib import Path
base_dir = os.path.dirname(__file__)

# instantiate objects for extraction, transformation and load
yahoo_fetcher = YahooFinanceFetcher()
pdr_fetcher = PDRFetcher()
ticker_extender = TickerExtender()

date_range = pd.date_range(start="2000-01-01", end=datetime.today(), freq="B")

seed =  8523


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
import numpy as np

one_hot_encoder = make_column_transformer(
                      (OneHotEncoder(sparse_output=False, drop='if_binary'),
                        make_column_selector(dtype_exclude=np.number)
                      ),
                      remainder="passthrough",
                      verbose_feature_names_out=False,
                  ).set_output(transform="pandas")

from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.preprocessing import RollingFeatures
from skforecast.utils import save_forecaster


# Functions
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
    yahoo_fetcher.fetch_save_prices_info_parallel(macro_list, save_path = Path(base_dir) / "data/extracted/macro")

    # pdr finance
    print("Extracting macroindicators from pdr...\n")
    pdr_fetcher.fetch_save_prices_info_parallel(macro_pdr_dict, dir_path = Path(base_dir) / "data/extracted/macro")

    # euroyield
    fetch_euro_yield(save_path = Path(base_dir) / "data/extracted/macro/eurostat_euro_yield.parquet")



def transform_tickers(forecast_horizon: int = 5):
    base_dir = os.path.dirname(__file__)

    print("Transforming...")
    print("Enriching ticker dataframes.")
    ticker_df_list = ticker_extender.transform_daily_tickers_parallel_experiment(Path(base_dir) / "data/extracted/OHLCV", forecast_horizon=forecast_horizon)

    print("Merging into one collection.")
    merged_ticker_df = ticker_extender.merge_tickers(ticker_df_list)

    merged_ticker_df.write_parquet(Path(base_dir) / "data/transformed/dataset.parquet")


def load_select_features():
    transformed_df = pl.read_parquet(Path(base_dir) / "data/transformed/dataset.parquet")


    data_long_indicators = transformed_df.select(pl.exclude(["high","low","open","volume","currency","industry","sector","region"])
                                                ).filter(pl.col("datetime") >= datetime(2012,6,1)) # Retain information from only 2012. This might not be necessary. To check.
                                                

    data_long_indicators_series = data_long_indicators.select(["datetime","close","symbol"])
    data_long_indicators_exog = data_long_indicators.select(pl.exclude("close"))

    # Select only necessary exog columns
    exog_selected = ["datetime","symbol","country", 'growth_adj_365d_lagged_5', 'SMA22_lagged_5', '30d_volatility_lagged_5', 
                                                    'volume_last_10_lagged_5', 'pct_3_month_high_lagged_5', 'pct_1_year_high_lagged_5', 'pct_6_month_low_lagged_5', 
                                                    '10_day_rolling_return_lagged_5', '22_day_rolling_return_lagged_5', '126_day_rolling_return_lagged_5', '66_day_rolling_sharpe_lagged_5', 
                                                    'autocorrelation_lag_1_lagged_5', 'autocorrelation_lag_6_lagged_5', 'rolling_skew_5_lagged_5', 'rolling_skew_10_lagged_5', 'rolling_skew_15_lagged_5']


    data_long_indicators_exog = data_long_indicators_exog.select(exog_selected)


    return data_long_indicators_series, data_long_indicators_exog


def generate_training_dicts(date_range, training_window, data_long_indicators_series, data_long_indicators_exog):

    date_range = pd.date_range(start="2000-01-01", end=datetime.today(), freq="B")

    start = datetime.strftime(date_range[-training_window], "%Y-%m-%d") # This in practice should fecth the data from S3 or the feature store and perform this
    end = datetime.strftime(date_range[-6], "%Y-%m-%d") # Also, it should be done on weekends, but right now it is not a weekend, so the last day available day is a Thursday

    series_dict, exog_dict, _, _, _= long_series_exog_to_dict(dataframe = data_long_indicators_series.to_pandas(), # Same with this, the df should be loaded from S3
                                    series_id_column = "symbol", 
                                    start_train = start,
                                    end_train = end,
                                    start_test = start,
                                    end_test = end,
                                    exog_dataframe = data_long_indicators_exog.to_pandas(),
                                    index_freq = "B",
                                    fill_nan = "ffill",
                                    partition_name="",
                                    verbose = True)
    
    return series_dict, exog_dict


def generate_out_of_sample_pred(date_range, training_window, transformer_exog, model):
    # For residuals we would need a train and val
    start_train = datetime.strftime(date_range[-training_window - 999], "%Y-%m-%d")
    end_train = datetime.strftime(date_range[- 999], "%Y-%m-%d")
    start_val = datetime.strftime(date_range[- 998], "%Y-%m-%d")
    end_val = datetime.strftime(date_range[-6], "%Y-%m-%d")

    series_dict_val, exog_dict_val, series_dict_train, exog_dict_train, series_dict_val_test = long_series_exog_to_dict(dataframe = data_long_indicators_series.to_pandas(), # Same with this, the df should be loaded from S3
                                    series_id_column = "symbol", 
                                    start_train = start_train,
                                    end_train = end_train,
                                    start_test = start_val,
                                    end_test = end_val,
                                    exog_dataframe = data_long_indicators_exog.to_pandas(),
                                    index_freq = "B",
                                    fill_nan = "ffill",
                                    partition_name="Validation",
                                    verbose = True)

    params = {
        "series_train": series_dict_train,
        "series": series_dict_val,
        "model": model,
        "forecast_horizon": 5,
        "exog_train": exog_dict_train,
        "exog": exog_dict_val,
        "window_stats": ["ratio_min_max","mean","mean"],
        "window_sizes": [2,5,15],
        "lags": [1, 6, 9],
        "differentiation": 1,
        "transformer_exog": transformer_exog,
        "n_jobs": 6,
        "refit": 1,
        "encoding": "onehot",
        "fixed_train_size": True,
        "suppress_warnings": False,
        "show_progress": True,
        "verbose": True
    }

    _, backtest_predictions, _ = evaluate_recursive_multiseries_2(**params)

    directory = Path(base_dir) / "data/predictions"
    os.makedirs(directory,exist_ok=True)
    pl.from_pandas(backtest_predictions).write_parquet(directory / "out_of_sample.parquet")

    return backtest_predictions, series_dict_val_test


def train(model, series_dict, exog_dict, out_sample_pred, out_sample_dict_test, transformer_exog):
    window_features = RollingFeatures(stats=["ratio_min_max","mean","mean"], window_sizes=[2,5,15])
    forecaster_recursive = ForecasterRecursiveMultiSeries(
                        regressor        = model,
                        lags             = [1, 6, 9],
                        window_features  = window_features,
                        differentiation  = 1,
                        differentiator   = "pct",
                        transformer_exog = one_hot_encoder,
                        encoding         = "onehot"
                    )

    forecaster_recursive.fit(series=series_dict, 
                            exog=exog_dict,
                                suppress_warnings=False, store_in_sample_residuals=False)
    

    # Store out-sample residuals in the forecaster
    # ==============================================================================
    y_pred_dict = {symbol: out_sample_pred[symbol] for symbol in out_sample_pred.keys()}

    forecaster_recursive.set_out_sample_residuals(
        y_true = out_sample_dict_test,
        y_pred = y_pred_dict
    ) 

    directory = Path(base_dir) / "model"
    os.makedirs(directory,exist_ok=True)
    print(directory)
    save_forecaster(
        forecaster_recursive, 
        file_name = 'model/forecaster_recursive_multiseries.joblib', 
        save_custom_functions = True, 
        verbose = False
    )

    print("That's all fellas!")

if __name__ == "__main__":

    # Extract
    extract_tickers()

    # Transform
    transform_tickers()

    # Load
    data_long_indicators_series, data_long_indicators_exog = load_select_features()

    # Generate dicts for training
    date_range = pd.date_range(start="2000-01-01", end=datetime.today(), freq="B")
    training_window = 2609
    series_dict, exog_dict = generate_training_dicts(date_range, training_window, data_long_indicators_series, data_long_indicators_exog)

    # Generate out of sample residuals
    model = HistGradientBoostingRegressor(random_state = seed, loss="quantile", quantile=0.5)

    out_sample_pred, out_sample_dict_test = generate_out_of_sample_pred(date_range, training_window, one_hot_encoder, model)


    train(model, series_dict, exog_dict, out_sample_pred, out_sample_dict_test,transformer_exog=one_hot_encoder)








