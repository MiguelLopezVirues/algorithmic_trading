from skforecast.preprocessing import series_long_to_dict, exog_long_to_dict

from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np

from typing import Dict, Union, List
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.preprocessing import RollingFeatures
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster_multiseries

import warnings
from skforecast.exceptions.exceptions import MissingValuesWarning


import matplotlib.pyplot as plt
import plotly.graph_objects as go

from typing import Optional

from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures

from statsmodels.tsa.stattools import ccf

from typing import Tuple

import exchange_calendars as ecals
import pandas_market_calendars as mcal

from sklearn.pipeline import make_pipeline


# General Libraries
from datetime import datetime
import math
import random
import sys
import warnings

# Data Handling and Processing
import pandas as pd
import polars as pl
import numpy as np
from typing import List, Union, Optional, Dict

# Scikit-learn and Forecasting
import sklearn
import skforecast
from skforecast.plot import plot_residuals, plot_prediction_distribution, set_dark_theme
from skforecast.recursive import ForecasterRecursiveMultiSeries, ForecasterRecursive
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import TimeSeriesFold, OneStepAheadFold, backtesting_forecaster, bayesian_search_forecaster, backtesting_forecaster_multiseries, bayesian_search_forecaster_multiseries
from skforecast.feature_selection import select_features_multiseries
from skforecast.preprocessing import RollingFeatures, series_long_to_dict, exog_long_to_dict
from skforecast.exceptions import OneStepAheadValidationWarning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor

# Feature Engineering
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures
from sklearn.pipeline import make_pipeline

# Plotting and Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import pacf, acf

# Time Series and Statistical Analysis
import networkx as nx



# Warnings Configuration
warnings.filterwarnings('once')
from skforecast.exceptions.exceptions import MissingValuesWarning, LongTrainingWarning, IgnoredArgumentWarning, MissingExogWarning
warnings.simplefilter('ignore', category=LongTrainingWarning)
warnings.simplefilter('ignore', category=IgnoredArgumentWarning)
warnings.simplefilter('ignore', category=MissingValuesWarning)
warnings.simplefilter('ignore', category=MissingExogWarning)

# Custom Modules
sys.path.append("..")
from src.support.data_transformation import TickerExtender, TechnicalIndicators, FileHandler
from src.support.timeseries_support import TimeSeriesAnalysis
from src.support.file_handling import FileHandler
import src.support.data_visualization as dv

# Instantiate objects
ticker_extender = TickerExtender()
file_handler = FileHandler()
import os
base_dir = os.path.dirname(__file__)
from pathlib import Path


from src.support.config.countries_exchanges import exchange_calendars_exog


def fill_na_dict(
                ts_dict: Dict[str, pd.Series], 
                method: str = "ffill", 
                verbose: bool = False, 
                series_name: str = "Series"
            ) -> Dict[str, pd.Series]:
    """
    Fills missing values in a dictionary of time series using the specified method.

    Args:
        ts_dict (Dict[str, pd.Series]): Dictionary where keys are series names and values are Pandas Series.
        method (str, optional): Method for filling NaN values. Options: "interpolate", "ffill", or None. Defaults to "ffill".
        verbose (bool, optional): If True, prints the method applied. Defaults to False.
        series_name (str, optional): Name of the series for logging purposes. Defaults to "Series".

    Returns:
        Dict[str, pd.Series]: Dictionary with NaN values handled according to the selected method.

    Raises:
        ValueError: If an unsupported method is specified.
    """
    if method == "interpolate":
        print(f"{series_name} values interpolated") if verbose else None
        ts_dict = {k: v.interpolate() for k, v in ts_dict.items()}

    elif method == "ffill":
        print(f"{series_name} values filled forward") if verbose else None
        ts_dict = {k: v.ffill() for k, v in ts_dict.items()}

    elif method == None:
        pass

    else:
        raise ValueError("NaN imputation method not implemented.")
    
    return ts_dict

def long_series_exog_to_dict(dataframe: pd.DataFrame,
                                series_id_column: str, 
                                start_train:str,
                                end_train:str,
                                start_test: str,
                                end_test:str,
                                exog_dataframe: Optional[pd.DataFrame] = None,
                                index_freq: str = "B",
                                fill_nan: str = "ffill",
                                verbose: bool = False,
                                partition_name: str = "Train") -> Dict[str, pd.DataFrame]:
    """
    Converts a long-format time series into a dictionary of train-test partitions.

    Args:
        dataframe (pd.DataFrame): Input DataFrame in long format with a 'datetime' index and 'close' column.
        series_id_column (str): Column name that identifies different time series.
        start_train (str): Start date for the training period.
        end_train (str): End date for the training period.
        start_test (str): Start date for the testing period.
        end_test (str): End date for the testing period.
        exog_dataframe (Optional[pd.DataFrame], optional): DataFrame with exogenous variables. Defaults to None.
        index_freq (str, optional): Frequency of the datetime index. Defaults to "B" (business days).
        fill_nan (str, optional): Method for handling NaN values. Defaults to "ffill".
        verbose (bool, optional): If True, prints processing details. Defaults to False.
        partition_name (str, optional): Name of the partition (e.g., "Train"). Defaults to "Train".

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with train and test partitions of the time series.

    Notes:
        - Filters out missing values due to closed markets.
        - Handles NaN values using the specified method.
    """
    # Filter warnings from closed market missing values
    warnings.simplefilter('ignore', category=MissingValuesWarning)

    # Transform series and exog to dictionaries
    # ==============================================================================
    series_dict = series_long_to_dict(
                    data      = dataframe,
                    series_id = series_id_column,
                    index     = 'datetime', # Add error handling for column 'datetime'
                    values    = 'close',
                    freq      = index_freq
                    )
    
    # Handle missing values in the series inside the dict
    series_dict = fill_na_dict(ts_dict = series_dict, 
                 method = fill_nan, 
                 verbose=verbose, 
                 series_name="Autoreg series")

    # Series train-test split
    # ==============================================================================
    series_dict_train = {k: v.loc[start_train: end_train,] for k, v in series_dict.items()}
    series_dict_test  = {k: v.loc[start_test:end_test,] for k, v in series_dict.items()} # To obtain out_of_sample residuals
    series_dict  = {k: v.loc[start_train:end_test,] for k, v in series_dict.items()} # Equivalent to test, must include train



    if isinstance(exog_dataframe, pd.DataFrame):
        # Transform exog to dictionaries
        # ==============================================================================
        exog_dict = exog_long_to_dict(
                        data      = exog_dataframe,
                        series_id = series_id_column,
                        index     = 'datetime', # Add error handling for column 'datetime'
                        freq      = index_freq
                    )
        
        exog_dict = fill_na_dict(ts_dict = exog_dict, 
                    method = fill_nan, verbose=verbose, series_name="Exog series")
    
        # Exog train-test splits
        # ==============================================================================
        exog_dict_train   = {k: v.loc[start_train: end_train,] for k, v in exog_dict.items()}
        exog_dict  = {k: v.loc[start_train:end_test,] for k, v in exog_dict.items()} # Equivalent to test, must include train

    else:
        exog_dict = None
        exog_dict_train = None

    # Series description
    # ==============================================================================
    if verbose:
        for k in series_dict.keys():
            print(f"{k}:")
            try:
                print(
                    f"\t{partition_name}: len={len(series_dict_train[k])}, {series_dict_train[k].index[0]}"
                    f" --- {series_dict_train[k].index[-1]} "
                    f" (len={len(series_dict_train[k])})"
                    f" (missing={series_dict_train[k].isnull().sum()})"
                    f" First day: {series_dict_train[k].index.min().day_name()}. Last day: {series_dict_train[k].index.max().day_name()}."
                )
            except:
                print(f"\t{partition_name}: len=0")
            try:

                print(
                    f"\tTest : len={len(series_dict_test[k])}, {series_dict_test[k].index[0]}"
                    f" --- {series_dict_test[k].index[-1]} "
                    f" (len={len(series_dict_test[k])})"
                    f" (missing={series_dict_test[k].isnull().sum()})"
                    f" First day: {series_dict_test[k].index.min().day_name()}. Last day: {series_dict_test[k].index.max().day_name()}."
                )
            except:
                print(f"\tTest : len=0")

        for k in exog_dict.keys():
            print(f"{k}:")
            try:
                print(
                    f"\t{partition_name}: len={len(exog_dict_train[k])}, {exog_dict_train[k].index[0]}"
                    f" --- {exog_dict_train[k].index[-1]} "
                    f" (len={len(exog_dict_train[k])})"
                    # f" (missing={exog_dict_train[k].isnull().sum()})"
                    f" First day: {exog_dict_train[k].index.min().day_name()}. Last day: {exog_dict_train[k].index.max().day_name()}."
                )
            except:
                print(f"\t{partition_name}: len=0")
            try:
                exog_dict_test  = {k: v.loc[start_test:end_test,] for k, v in exog_dict.items()} # Just for verbose purposes
                print(
                    f"\tTest : len={len(exog_dict_test[k])}, {exog_dict_test[k].index[0]}"
                    f" --- {exog_dict_test[k].index[-1]} "
                    f" (len={len(exog_dict_test[k])})"
                    # f" (missing={exog_dict_test[k].isnull().sum().sum()})"
                    f" First day: {exog_dict_test[k].index.min().day_name()}. Last day: {exog_dict_test[k].index.max().day_name()}."
                )
            except:
                print(f"\tTest : len=0")

    
    return series_dict, exog_dict, series_dict_train, exog_dict_train, series_dict_test


def evaluate_recursive_multiseries(
                                series_train: Union[Dict, pd.DataFrame],
                                series: Union[Dict, pd.DataFrame],
                                model: object, 
                                forecast_horizon: int, 
                                window_stats: List[str], 
                                window_sizes: List[int], 
                                lags: List[int],
                                exog_train: Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]] = None,
                                exog: Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]] = None,
                                transformer_exog: Optional[object] = None,
                                encoding: str = "onehot",
                                differentiation: int = 1, 
                                refit: Union[bool,int] = 8,
                                fixed_train_size: bool = True,
                                suppress_warnings: bool = False,
                                show_progress: bool = True,
                                n_jobs: Union[str,int] = "auto",
                                verbose: bool = False,
                                metric: str = "mean_absolute_percentage_error",
                                best_params: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, ForecasterRecursiveMultiSeries]:
    """
    Evaluates a recursive multi-series forecasting model using time series backtesting.

    Args:
        series_train (Union[Dict[str, pd.Series], pd.DataFrame]): Training time series data.
        series (Union[Dict[str, pd.Series], pd.DataFrame]): Full time series data for evaluation.
        model (object): Regression model to be used in forecasting.
        forecast_horizon (int): Number of steps ahead to forecast.
        window_stats (List[str]): Statistical features for rolling window.
        window_sizes (List[int]): Sizes of rolling windows for feature extraction.
        lags (List[int]): Lags to be used as predictors.
        exog_train (Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]], optional): Exogenous variables for training. Defaults to None.
        exog (Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]], optional): Exogenous variables for evaluation. Defaults to None.
        transformer_exog (Optional[object], optional): Transformer for exogenous variables. Defaults to None.
        encoding (str, optional): Encoding type for categorical variables. Defaults to "onehot".
        differentiation (int, optional): Degree of differentiation applied to the series. Defaults to 1.
        refit (Union[bool, int], optional): Refit strategy for the model during backtesting. Defaults to 8.
        fixed_train_size (bool, optional): Whether to keep training size fixed in backtesting. Defaults to True.
        suppress_warnings (bool, optional): Whether to suppress warnings. Defaults to False.
        show_progress (bool, optional): Whether to show progress bar during backtesting. Defaults to True.
        n_jobs (Union[str, int], optional): Number of parallel jobs for execution. Defaults to "auto".
        verbose (bool, optional): Whether to print debugging information. Defaults to False.
        metric (str, optional): Performance metric to evaluate the model. Defaults to "mean_absolute_percentage_error".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, ForecasterRecursiveMultiSeries]: 
            - metrics_levels: DataFrame with evaluation metrics per series.
            - backtest_predictions: DataFrame with backtesting predictions.
            - forecaster_recursive: Trained ForecasterRecursiveMultiSeries object.
    """

    # Define rolling features
    window_features = RollingFeatures(stats=window_stats, window_sizes=window_sizes)

    # Fit forecaster
    # ==============================================================================
    # Define the forecaster. 
    forecaster_recursive = ForecasterRecursiveMultiSeries(
                        regressor        = model,
                        lags             = lags,
                        window_features  = window_features,
                        differentiation  = differentiation,
                        differentiator   = "pct",
                        transformer_exog = transformer_exog,
                        encoding         = encoding,
                    )
    
    if best_params:
        forecaster_recursive.set_params(**best_params)

    # Fit
    forecaster_recursive.fit(series=series_train, 
                            exog=exog_train,
                                suppress_warnings=suppress_warnings)
    
    

    # Backtesting
    # ==============================================================================
    
    # Define cross-validation scheme
    cv = TimeSeriesFold(
            steps              = forecast_horizon,
            initial_train_size = len(next(iter(series_train.values()))) if isinstance(series_train, dict) else series_train.shape[0], # get first key to obtain length of series
            refit              = refit,
            fixed_train_size   = fixed_train_size,
            differentiation    = differentiation
        )
    
    # Perform backtesting
    metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
        forecaster            = forecaster_recursive,
        series                = series,
        exog                  = exog,
        cv                    = cv,
        metric                = metric,
        add_aggregated_metric = False,
        n_jobs                = n_jobs,
        verbose               = verbose,
        show_progress         = show_progress,
        suppress_warnings     = suppress_warnings,
        use_in_sample_residuals = False 
    )


    print(f"The mean {metric} is {metrics_levels[metric].mean()}")

    return metrics_levels, backtest_predictions, forecaster_recursive

# def evaluate_recursive_multiseries_2(
#                                 series_train: Union[Dict, pd.DataFrame],
#                                 series: Union[Dict, pd.DataFrame],
#                                 model: object, 
#                                 forecast_horizon: int, 
#                                 window_stats: List[str], 
#                                 window_sizes: List[int], 
#                                 lags: List[int],
#                                 exog_train: Union[Dict, pd.DataFrame] = None,
#                                 exog: Union[Dict, pd.DataFrame] = None,
#                                 transformer_exog: object = None,
#                                 encoding: str = "onehot",
#                                 differentiation: int = 1, 
#                                 refit: Union[bool,int] = 8,
#                                 fixed_train_size: bool = True,
#                                 suppress_warnings: bool = False,
#                                 show_progress: bool = True,
#                                 n_jobs: Union[str,int] = "auto",
#                                 verbose: bool = False):

#     # Fit forecaster
#     # ==============================================================================
#     # Define the forecaster. 
#     window_features = RollingFeatures(stats=window_stats, window_sizes=window_sizes)
#     forecaster_recursive = ForecasterRecursiveMultiSeries(
#                         regressor        = model,
#                         lags             = lags,
#                         window_features  = window_features,
#                         differentiation  = differentiation,
#                         differentiator   = "pct",
#                         transformer_exog = transformer_exog,
#                         encoding         = encoding,
#                     )
    
#     forecaster_recursive.fit(series=series_train, 
#                             exog=exog_train,
#                                 suppress_warnings=suppress_warnings)
    
#     display(forecaster_recursive)
    

#     # Backtesting
#     # ==============================================================================
#     cv = TimeSeriesFold(
#             steps              = forecast_horizon,
#             initial_train_size = len(next(iter(series_train.values()))) if isinstance(series_train, dict) else series_train.shape[0], # get first key to obtain length of series
#             refit              = refit,
#             fixed_train_size   = fixed_train_size,
#             differentiation    = differentiation
#         )
    

#     metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
#         forecaster            = forecaster_recursive,
#         series                = series,
#         exog                  = exog,
#         cv                    = cv,
#         metric                = "mean_absolute_percentage_error",
#         add_aggregated_metric = False,
#         n_jobs                = n_jobs,
#         verbose               = verbose,
#         show_progress         = show_progress,
#         suppress_warnings     = suppress_warnings
#     )


#     print(f"The mean MAPE is {metrics_levels['mean_absolute_percentage_error'].mean()}")

#     return metrics_levels, backtest_predictions, forecaster_recursive


def evaluate_recursive_multiseries_separate(dataframe: pd.DataFrame,
                                series_id_column: str, 
                                start_train:str,
                                end_train:str,
                                start_test: str,
                                end_test:str,
                                model: object, 
                                forecast_horizon: int, 
                                window_stats: List[str], 
                                window_sizes: List[int], 
                                lags: List[int],
                                exog_dataframe: pd.DataFrame = None,
                                transformer_exog: object = None,
                                encoding: str = "onehot",
                                index_freq: str = "B",
                                fill_nan: str = "ffill",
                                differentiation: int = 1, 
                                refit: Union[bool,int] = 8,
                                fixed_train_size: bool = True,
                                suppress_warnings: bool = False,
                                show_progress: bool = True,
                                n_jobs: Union[str,int] = "auto",
                                verbose: bool = False):

    series_dict, exog_dict, series_dict_train, exog_dict_train, _ = long_series_exog_to_dict(dataframe = dataframe,
                                                                                            series_id_column = series_id_column, 
                                                                                            start_train = start_train,
                                                                                            end_train = end_train,
                                                                                            start_test = start_test,
                                                                                            end_test = end_test,
                                                                                            exog_dataframe = exog_dataframe,
                                                                                            index_freq = index_freq,
                                                                                            fill_nan = fill_nan,
                                                                                            verbose = verbose)
    

    metrics_levels, backtest_predictions, forecaster_recursive = evaluate_recursive_multiseries_2(series_train=series_dict_train,
                                                                                                series=series_dict,
                                                                                                model=model, 
                                                                                                forecast_horizon=forecast_horizon, 
                                                                                                window_stats=window_stats, 
                                                                                                window_sizes=window_sizes, 
                                                                                                lags=lags,
                                                                                                exog_train=exog_dict_train,
                                                                                                exog=exog_dict,
                                                                                                transformer_exog=transformer_exog,
                                                                                                encoding=encoding,
                                                                                                differentiation=differentiation, 
                                                                                                refit=refit,
                                                                                                fixed_train_size=fixed_train_size,
                                                                                                suppress_warnings=suppress_warnings,
                                                                                                show_progress=show_progress,
                                                                                                n_jobs=n_jobs,
                                                                                                verbose=verbose)



    return metrics_levels, backtest_predictions, forecaster_recursive, series_dict_train, exog_dict_train


def evaluate_direct_multiseries(dataframe: pd.DataFrame,
                                start_train:str,
                                end_val:str,
                                end_test:str,
                                model: object, 
                                forecast_horizon:int, 
                                target_list: List[str], 
                                predictors_list: List[str], 
                                window_stats: List[str], 
                                window_sizes: List[int], 
                                lags: List[int],
                                exog_features: List[str] = [], 
                                differentiation: int = 1, 
                                refit: Union[bool, int]= 24,
                                fixed_train_size: int= True,
                                verbose: bool = False,
                                suppress_warnings: bool = True,
                                show_individual_progress = False,
                                n_jobs: Union[str, int] = "auto"):
    # Entrenar y realizar backtesting de un modelo para cada item - direct
    # ======================================================================================
    items = []
    mape_values = []
    predictions = []
    forecasters_direct = {} # Should be aligned for predictions to start on Mondays and end on Fridays

    initial_train_size = len(dataframe.loc[start_train:end_val,])
    initial_train_start = dataframe.loc[[start_train],].index.day_name()[0]
    initial_train_end = dataframe.loc[[end_val],].index.day_name()[0]
    print(f"Initial train size starts on {start_train} ({initial_train_start}) and ends on {end_val} ({initial_train_end}).")

    for item in tqdm(target_list):
        # warnings.simplefilter('ignore', category='LongTrainingWarning')
        # Definir el forecaster
        window_features = RollingFeatures(stats=window_stats*len(window_sizes), window_sizes=window_sizes)

        forecasters_direct[item] = ForecasterDirectMultiVariate(
                                        level                   = item,
                                        regressor               = model,
                                        steps                   = forecast_horizon,
                                        lags                    = lags,
                                        window_features         = window_features,
                                        differentiation         = differentiation,
                                        differentiator= "pct",
                                        n_jobs=n_jobs
                                        )
        
        # Backtesitng 
        cv = TimeSeriesFold(
                    steps              = forecast_horizon,
                    initial_train_size = initial_train_size,
                    refit              = refit,
                    fixed_train_size   = fixed_train_size,
                    differentiation=1
                )

        metric, preds = backtesting_forecaster_multiseries(
                                            forecaster            = forecasters_direct[item],
                                            series                = dataframe.loc[start_train:end_test,predictors_list], 
                                            levels                = [item],
                                            exog                  = dataframe.loc[start_train:end_test,exog_features] if exog_features else None,
                                            cv                    = cv,
                                            metric                = 'mean_absolute_percentage_error',
                                            add_aggregated_metric = False,
                                            verbose               = verbose,
                                            suppress_warnings     = suppress_warnings,
                                            show_progress         = show_individual_progress ,
                                            n_jobs                = n_jobs
                                        )   

        
        items.append(item)
        mape_values.append(metric.at[0, 'mean_absolute_percentage_error'])
        predictions.append(preds)
        

    # Resultados
    direct_series_results = pd.DataFrame({
                        "levels": items,
                        "mape": mape_values,
                        "predictions": predictions
                        })
    
    print(f"The mean MAPE for uniseries is {direct_series_results['mape'].mean():.6f}")
    
    return direct_series_results, forecasters_direct

def evaluate_recursive_multiseries_dataframe(dataframe: pd.DataFrame,
                                start_train:str,
                                end_val:str,
                                end_test:str,
                                model: object, 
                                forecast_horizon:int, 
                                target_list: List[str], 
                                predictors_list: List[str], 
                                window_stats: List[str], 
                                window_sizes: List[int], 
                                lags: List[int],
                                exog_features: List[str] = [], 
                                differentiation: int = 1, 
                                refit: Union[bool, int]= 8,
                                fixed_train_size: int= True,
                                verbose: bool = False,
                                suppress_warnings: bool = True,
                                show_individual_progress = False,
                                show_progress: bool = False,
                                n_jobs: Union[str, int] = "auto"):


  # Fit forecaster
  # ==============================================================================
  # Define the forecaster. 
  window_features = RollingFeatures(stats=window_stats, window_sizes=window_sizes)
  forecaster_recursive = ForecasterRecursiveMultiSeries(
                      regressor       = model,
                      lags            = lags,
                      window_features = window_features,
                      differentiation = differentiation,
                      differentiator="pct"
                  )
  
  forecaster_recursive.fit(series            = dataframe.loc[start_train:end_val,predictors_list], 
                           exog              = dataframe.loc[start_train:end_val,exog_features] if exog_features else None,
                           suppress_warnings = suppress_warnings)
  

  # Backtesting
  # ==============================================================================
  cv = TimeSeriesFold(
          steps              = forecast_horizon,
          initial_train_size = dataframe.loc[start_train:end_val,].shape[0], # get first key to obtain length of series
          refit              = refit,
          fixed_train_size   = fixed_train_size,
          differentiation    = differentiation
       )
  

  metrics_levels, backtest_predictions = backtesting_forecaster_multiseries(
      forecaster            = forecaster_recursive,
      series                = dataframe.loc[start_train:end_test, predictors_list],
      exog                  = dataframe.loc[start_train:end_test, exog_features] if exog_features else None,
      levels                = target_list,
      cv                    = cv,
      metric                = "mean_absolute_percentage_error",
      add_aggregated_metric = False,
      n_jobs                = n_jobs,
      verbose               = verbose,
      show_progress         = show_progress,
      suppress_warnings     = suppress_warnings
  )



  print(f"The mean MAPE is {metrics_levels['mean_absolute_percentage_error'].mean()}")

  return metrics_levels, backtest_predictions, forecaster_recursive


def get_opening_days_market(ecal_exchange_symbol: str, start: str, end: str):
    exchange_calendar = ecals.get_calendar(ecal_exchange_symbol)

    # Get the market schedule
    exchange_schedule = exchange_calendar.schedule.loc[start:end].index.to_frame(index=False, name="datetime")

    # Assign open days
    exchange_schedule = exchange_schedule.assign(market_open=1).set_index("datetime").asfreq("B").fillna(0)

    return exchange_schedule


def get_calendar_cyclical_features(dataframe: pd.DataFrame,
                                   datetime_column: str = "index",
                                   drop_original: bool = True) -> pd.DataFrame:

    original_columns = dataframe.columns
    # Calendar features
    # ==============================================================================
    features_to_extract = [
        'year',
        'month',
        'week',
        'day_of_month',
        'day_of_week'
    ]
    
    calendar_transformer = DatetimeFeatures(
        variables           = datetime_column,
        features_to_extract = features_to_extract,
        drop_original       = False
    )

    # Cliclical encoding of calendar features
    # ==============================================================================
    features_to_encode = [
        "month",
        "week",
        "day_of_month",
        "day_of_week"
    ]
    max_values = {
        "month": 12,
        "week": 52,
        "day_of_month": 30,
        "day_of_week": 6
    }

    cyclical_encoder = CyclicalFeatures(
                            variables     = features_to_encode,
                            max_values    = max_values,
                            drop_original = drop_original
                    )

    exog_transformer = make_pipeline(
                            calendar_transformer,
                            cyclical_encoder
                    )

    dataframe_plus_calendar_cyclical = exog_transformer.fit_transform(dataframe)

    # exog_columns = dataframe_plus_calendar_cyclical.columns.difference(original_columns).tolist()
    
    return dataframe_plus_calendar_cyclical



def get_opening_days_market(ecal_exchange_symbol: str, start: str, end: str):
    exchange_calendar = ecals.get_calendar(ecal_exchange_symbol)

    # Get the market schedule
    exchange_schedule = exchange_calendar.schedule.loc[start:end].index.to_frame(index=False, name="datetime")

    # Assign open days
    exchange_schedule["market_open"] = 1 
    exchange_schedule = exchange_schedule.set_index("datetime").asfreq("B").tz_localize(None).fillna(0)

    return exchange_schedule

def get_opening_days_market_pandas(mcal_exchange_symbol: str, start: str, end: str):
    # Get calendar
    exchange_calendar = mcal.get_calendar(mcal_exchange_symbol)

    # Get the market schedule
    exchange_schedule = exchange_calendar.valid_days(start_date=start, end_date=end)
    exchange_schedule = exchange_schedule.to_frame(index=False, name='datetime')   

    # Assign open days
    exchange_schedule["market_open"] = 1
    exchange_schedule = exchange_schedule.set_index("datetime").asfreq("B").tz_localize(None).fillna(0)

    return exchange_schedule



def get_stocks_info():
    data_dict = {
        'symbol': [],
        'industry': [],
        'sector': [],
        'country': [],
        'region': []
    }

    for file in file_handler.list_all_files(Path(base_dir) / "../../data/extracted/OHLCV/"):
        if "parquet" in str(file):
            stock_first_row = file_handler.read_parquet_file(file)[0]
            data_dict['symbol'].append(stock_first_row['symbol'][0])
            data_dict['industry'].append(stock_first_row['industry'][0])
            data_dict['sector'].append(stock_first_row['sector'][0])
            data_dict['country'].append(stock_first_row['country'][0])
            data_dict['region'].append(stock_first_row['region'][0])

    stocks_info = pl.DataFrame(data_dict)

    return stocks_info



def get_opening_days_market_exog(dataframe: pd.DataFrame, symbol_column: str = "symbol"):

    stocks_info = get_stocks_info().to_pandas()

    open_market_df = pd.DataFrame()
    for i, symbol in enumerate(dataframe[symbol_column].unique()):
        # Get country and region
        country = stocks_info.loc[stocks_info["symbol"] == symbol, "country"].iloc[0].title().replace("_"," ")
        region = stocks_info.loc[stocks_info["symbol"] == symbol, "region"].iloc[0]

        # Get the market exchange symbol
        exchange_symbol = exchange_calendars_exog[region][country]

        # Define start and end for the calendar dates
        start = dataframe.loc[dataframe["symbol"] == symbol].index.min()
        end = dataframe.loc[dataframe["symbol"] == symbol].index.max()


        df = pd.concat([dataframe.loc[dataframe["symbol"] == symbol], get_opening_days_market_pandas(exchange_symbol, start=start, end=end)], axis=1)

        open_market_df = pd.concat([open_market_df,df],axis=0)   

    return open_market_df




def plot_ccf_symmetric_statsmodels(x, y, max_lag=30, x_name="x", y_name="y", figsize=(10,5)):
    """
    Plot a symmetrical cross-correlation function (CCF) for two time series x and y
    using statsmodels.tsa.stattools.ccf, covering negative and positive integer lags.

    Parameters
    ----------
    x, y : pandas Series
        Time series (preferably stationary, e.g. returns) to be cross-correlated.
        Should share a common time index or at least significant overlap.
    max_lag : int
        Maximum positive/negative lag to plot.
    x_name, y_name : str
        Names for the labels in the plot.
    figsize : tuple
        Size of the figure.

    Notes
    -----
    - statsmodels.ccf() only returns CCF for lags >= 0.
      To get negative lags, we swap x and y and combine the results.
    - This approach is an approximation if your series are autocorrelated.
      For real statistical testing, you might need more advanced methods.
    """
    # 1) Align on the common index to avoid misalignment
    x = x.dropna()
    y = y.dropna()
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx]
    y = y.loc[common_idx]

    # 2) Convert to numpy arrays (statsmodels ccf works on arrays)
    #    Optionally do differencing or log returns if the data isn't stationary
    #    Also recommended to standardize/normalize if you want 'similar scales'
    x_arr = x.values
    y_arr = y.values

    # 3) statsmodels' ccf => correlation for lags = 0..(len(x)-1)
    #    ccf(x, y): how y "follows" x
    #    ccf(y, x): how x "follows" y
    ccf_xy = ccf(x_arr, y_arr)  # This includes lag=0.. up to max possible
    ccf_yx = ccf(y_arr, x_arr)

    # 4) Truncate to max_lag + 1 terms from each (0..max_lag)
    #    ccf_xy[0] = lag 0, ccf_xy[1] = lag 1, ...
    #    so ccf_xy[:max_lag+1] means lags 0..max_lag
    ccf_xy_pos_lags = ccf_xy[: max_lag + 1]  # Lags: 0..+max_lag
    ccf_yx_pos_lags = ccf_yx[: max_lag + 1]  # Lags: 0..+max_lag

    # 5) To form negative lags, we can interpret ccf(y,x) at lag k
    #    as ccf(x,y) at lag -k. So ccf_yx_pos_lags[1..max_lag] => negative lags
    #    We'll reverse them to go from -max_lag..-1.
    neg_lags = np.arange(-max_lag, 0)
    pos_lags = np.arange(0, max_lag + 1)

    # slice [1:] to skip the zero-lag, then reverse
    ccf_neg = ccf_yx_pos_lags[1:][::-1]  # Lags: (max_lag..1) reversed => -1..-max_lag
    ccf_pos = ccf_xy_pos_lags            # Lags: 0..+max_lag

    # Combine them for symmetrical range
    ccf_full = np.concatenate([ccf_neg, ccf_pos])
    lags_full = np.concatenate([neg_lags, pos_lags])

    # 6) Rough 95% confidence intervals => ±2/sqrt(N)
    #    (N is effectively the sample size. This is a rule of thumb for white noise.)
    N = len(x_arr)
    conf = 2.0 / np.sqrt(N)

    # 7) Plot
    plt.figure(figsize=figsize)
    plt.stem(lags_full, ccf_full, basefmt=" ")
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(conf, color='red', linestyle='--', alpha=0.7, label='±95% conf')
    plt.axhline(-conf, color='red', linestyle='--', alpha=0.7)
    plt.title(f"Symmetric Cross-Correlation: {x_name} vs. {y_name}")
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True, ls=':')
    plt.show()

def calculate_top_pacf_df(df, cols, n_lags = 90):
    
    n_rows=math.ceil(len(cols)/2)
    n_cols = 1 if len(cols) == 1 else 2

    pacf_df = []
    fig, axs = plt.subplots(n_rows, 2, figsize=(12, 1.5*n_rows))
    axs = axs.flat
        
    for i, col in enumerate(cols):
        pacf_values = pacf(df[col].pct_change(1).dropna(), nlags=n_lags)
        pacf_values = pd.DataFrame({
            'lag': range(1, len(pacf_values)),
            'value': pacf_values[1:],
            'variable': col
        })
        pacf_df.append(pacf_values)
        plot_pacf(df[col].pct_change(1).dropna(), lags=n_lags, ax=axs[i])
        axs[i].set_title(col, fontsize=10)
        axs[i].set_ylim(-0.5, 1.1)
    plt.tight_layout()

    top_pacf_df = calculate_top_pacf(pacf_df, top_n_lags=5)

    return top_pacf_df

def calculate_top_pacf(pacf_df, top_n_lags=5):
    print("Showing all possible steps for window features:")
    print("=========")
    top_lags_pacf = set()
    top_lags_pacf_dict = dict()
    for pacf_values in pacf_df:
        variable = pacf_values['variable'].iloc[0]
        pacf_values['value'] = pacf_values['value'].abs()
        lags = pacf_values.nlargest(top_n_lags, 'value')['lag'].tolist()
        top_lags_pacf_dict[variable] = lags
        top_lags_pacf.update(lags)
        print(f"{variable}: {lags}")
    top_lags_pacf = list(top_lags_pacf)
    return top_lags_pacf


# def calculate_top_acf_pacf_df(df, cols, n_lags=90, method='pacf'):
    
#     n_rows = math.ceil(len(cols) / 2)
#     n_cols = 1 if len(cols) == 1 else 2

#     acf_pacf_df = []
#     fig, axs = plt.subplots(n_rows, 2, figsize=(12, 1.5 * n_rows))
#     axs = axs.flat
        
#     for i, col in enumerate(cols):
#         series = df[col].pct_change(1).dropna()
        
#         if method == 'pacf':
#             values = pacf(series, nlags=n_lags)
#             plot_func = plot_pacf

#         elif method == 'acf':
#             values = acf(series, nlags=n_lags)
#             plot_func = plot_acf
#         else:
#             raise ValueError("Method must be either 'pacf' or 'acf'")
        
#         values_df = pd.DataFrame({
#             'lag': range(1, len(values)),
#             'value': values[1:],
#             'variable': col
#         })
#         acf_pacf_df.append(values_df)
        
#         plot_func(series, lags=n_lags, ax=axs[i])
#         axs[i].set_title(col, fontsize=10)
#         axs[i].set_ylim(-0.5, 1.1)
    
#     plt.tight_layout()

#     top_acf_pacf_df = calculate_top_lags(acf_pacf_df, top_n_lags=5)

#     return top_acf_pacf_df

def calculate_top_acf_pacf_df(df, cols, n_lags=60, method='pacf'):
    
    n_rows = math.ceil(len(cols) / 2)
    n_cols = 1 if len(cols) == 1 else 2

    acf_pacf_df = []

    fig, axs = plt.subplots(n_rows, 2, figsize=(12, 1.5 * n_rows))
    axs = axs.flat
        
    for i, col in enumerate(cols):
        series = df[col].pct_change(1).dropna()
        
        if method == 'pacf':
            values = pacf(series, nlags=n_lags)
            plot_func = plot_pacf

        elif method == 'acf':
            values = acf(series, nlags=n_lags)
            plot_func = plot_acf
        else:
            raise ValueError("Method must be either 'pacf' or 'acf'")
        
        values_df = pd.DataFrame({
            'lag': range(1, len(values)),
            'value': values[1:],
            'variable': col
        })
        acf_pacf_df.append(values_df)
        
        plot_func(series, lags=n_lags, ax=axs[i])
        axs[i].set_title(col, fontsize=10)
        axs[i].set_ylim(-0.5, 1.1)
    
    plt.tight_layout()

    top_acf_pacf_df, top_acf_pacf_dict = calculate_top_lags(acf_pacf_df, top_n_lags=5)

    return top_acf_pacf_df, top_acf_pacf_dict



# def calculate_top_lags(acf_pacf_df, top_n_lags=5):
#     print("Showing all possible lags:")
#     print("=========")
#     top_lags = set()
#     top_lags_dict = dict()
#     for values_df in acf_pacf_df:
#         variable = values_df['variable'].iloc[0]
#         values_df['value'] = values_df['value'].abs()
#         lags = values_df.nlargest(top_n_lags, 'value')['lag'].tolist()
#         top_lags_dict[variable] = lags
#         top_lags.update(lags)
#         print(f"{variable}: {lags}")
#     top_lags = list(top_lags)
#     return top_lags


def calculate_top_lags(acf_pacf_df, top_n_lags=5):
    print("Showing all possible lags:")
    print("=========")
    top_lags = set()
    top_lags_dict = dict()
    for values_df in acf_pacf_df:
        variable = values_df['variable'].iloc[0]
        values_df['value'] = values_df['value'].abs()
        lags = values_df.nlargest(top_n_lags, 'value')['lag'].tolist()
        top_lags_dict[variable] = lags
        top_lags.update(lags)
        print(f"{variable}: {lags}")
    top_lags = list(top_lags)
    return top_lags, top_lags_dict


def plot_predicted_intervals(
    predictions: pd.DataFrame,
    y_true: pd.DataFrame,
    target_variable: str,
    initial_x_zoom: list=None,
    title: str=None,
    xaxis_title: str=None,
    yaxis_title: str=None,
):
    """
    Plot predicted intervals vs real values

    Parameters
    ----------
    predictions : pandas DataFrame
        Predicted values and intervals.
    y_true : pandas DataFrame
        Real values of target variable.
    target_variable : str
        Name of target variable.
    initial_x_zoom : list, default `None`
        Initial zoom of x-axis, by default None.
    title : str, default `None`
        Title of the plot, by default None.
    xaxis_title : str, default `None`
        Title of x-axis, by default None.
    yaxis_title : str, default `None`
        Title of y-axis, by default None.
    
    """

    fig = go.Figure([
        go.Scatter(name='Prediction', x=predictions.index, y=predictions[target_variable], mode='lines'),
        go.Scatter(name='Real value', x=y_true.index, y=y_true[target_variable], mode='lines'),
        go.Scatter(
            name='Upper Bound', x=predictions.index, y=predictions[f'{target_variable}_upper_bound'],
            mode='lines', marker=dict(color="#444"), line=dict(width=0), showlegend=False
        ),
        go.Scatter(
            name='Lower Bound', x=predictions.index, y=predictions[f'{target_variable}_lower_bound'],
            marker=dict(color="#444"), line=dict(width=0), mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)', fill='tonexty', showlegend=False
        )
    ])
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=800,
        height=400,
        margin=dict(l=20, r=20, t=35, b=20),
        hovermode="x",
        xaxis=dict(range=initial_x_zoom),
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0.001)
    )
    fig.show()
    
def plot_predicted_intervals_multiple_bounds(
    predictions: pd.DataFrame,
    y_true: pd.DataFrame,
    target_variable: str,
    initial_x_zoom: list=None,
    title: str=None,
    xaxis_title: str=None,
    yaxis_title: str=None,
    num_bounds: int = 1
):
    """
    Plot predicted intervals vs real values with multiple stacked bounds.

    Parameters
    ----------
    predictions : pandas DataFrame
        Predicted values and intervals.
    y_true : pandas DataFrame
        Real values of target variable.
    target_variable : str
        Name of target variable.
    initial_x_zoom : list, default `None`
        Initial zoom of x-axis, by default None.
    title : str, default `None`
        Title of the plot, by default None.
    xaxis_title : str, default `None`
        Title of x-axis, by default None.
    yaxis_title : str, default `None`
        Title of y-axis, by default None.
    num_bounds : int, default 1
        The number of bounds to plot stacked, by default 1.
    
    """
    # Initialize figure
    fig = go.Figure([
        # Plotting the predicted values
        go.Scatter(name='Prediction', x=predictions.index, y=predictions[target_variable], mode='lines'),
        # Plotting the real values
        go.Scatter(name='Real value', x=y_true.index, y=y_true[target_variable], mode='lines'),
    ])
    
    # Add stacked bounds
    for i in range(1, num_bounds + 1):
        upper_bound_col = f'{target_variable}_upper_bound_{i}'
        lower_bound_col = f'{target_variable}_lower_bound_{i}'
        
        if upper_bound_col in predictions.columns and lower_bound_col in predictions.columns:
            fig.add_trace(
                go.Scatter(
                    name=f'Upper Bound {i}', x=predictions.index, y=predictions[upper_bound_col],
                    mode='lines', marker=dict(color="#444"), line=dict(width=0), showlegend=False
                )
            )
            fig.add_trace(
                go.Scatter(
                    name=f'Lower Bound {i}', x=predictions.index, y=predictions[lower_bound_col],
                    marker=dict(color="#444"), line=dict(width=0), mode='lines',
                    fillcolor=f'rgba(68, 68, 68, {0.3 * i})', fill='tonexty', showlegend=False
                )
            )
    
    # Update the layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=800,
        height=400,
        margin=dict(l=20, r=20, t=35, b=20),
        hovermode="x",
        xaxis=dict(range=initial_x_zoom),
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0.001)
    )
    
    # Show the plot
    fig.show()


def empirical_coverage(y, lower_bound, upper_bound):
    """
    Calculate coverage of a given prediction interval's lower and upper bound
    """
    return np.mean(np.logical_and(y >= lower_bound, y <= upper_bound))


def plot_interval_coverage(interval_predictions: pd.DataFrame, series_dict_true: Dict, symbol, zoom_start: str, zoom_end: str):
    """
    Plots the predicted intervals for a given symbol and calculates the empirical coverage 
    of the prediction intervals.

    Parameters
    ----------
    interval_predictions : pd.DataFrame
        DataFrame containing the predicted intervals. It should have columns for the 
        lower and upper bounds of the interval.
    series_dict_true : Dict
        Dictionary containing the true time series data for multiple symbols.
    symbol : str
        Symbol representing the specific time series to analyze.
    zoom_start : str
        Start date for zooming into the plot (format: 'YYYY-MM-DD').
    zoom_end : str
        End date for zooming into the plot (format: 'YYYY-MM-DD').

    Returns
    -------
    tuple[float, float]
        - The empirical coverage of the predicted interval as a percentage.
        - The total area of the interval (sum of differences between upper and lower bounds).

    Notes
    -----
    - The function plots the real vs. predicted values within the specified zoom range.
    - The empirical coverage represents the percentage of actual values that fall within 
      the predicted interval.
    - The area of the interval is computed as the sum of the differences between the 
      upper and lower bounds.
    """
    # Plot intervals with zoom
    # ==============================================================================
    plot_predicted_intervals(
        predictions     = interval_predictions[[col for col in interval_predictions.columns if symbol in col]],
        y_true          = pd.DataFrame(series_dict_true[symbol].loc[interval_predictions.index], columns=[symbol]),
        target_variable = symbol,
        initial_x_zoom  = [zoom_start, zoom_end],
        title           = "Real value vs predicted in test data",
        xaxis_title     = "Date time",
        yaxis_title     = "users",
    )


    # Predicted interval coverage (on test data)
    # ==============================================================================
    coverage = empirical_coverage(
                    y           = series_dict_true[symbol].loc[interval_predictions.index.min():], # y_true
                    lower_bound = interval_predictions[f"{symbol}_lower_bound"], # predicted lower_bound
                    upper_bound = interval_predictions[f"{symbol}_upper_bound"] # predicted upper_bound
                )
    print(f"Predicted interval coverage: {round(100 * coverage, 2)} %")

    # Area of the interval
    # ==============================================================================
    area = (interval_predictions[f"{symbol}_upper_bound"] - interval_predictions[f"{symbol}_lower_bound"]).sum()
    print(f"Area of the interval: {round(area, 2)}")

    return round(100 * coverage, 2), area



 