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

def fill_na_dict(ts_dict: Dict, method: str = "ffill", verbose: bool = False, series_name: str = "Series"):
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
                                exog_dataframe: pd.DataFrame = None,
                                index_freq: str = "B",
                                fill_nan: str = "ffill",
                                verbose: bool = False,
                                partition_name: str = "Train"):
    
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
    
  
    series_dict = fill_na_dict(ts_dict = series_dict, 
                 method = fill_nan, verbose=verbose, series_name="Autoreg series")

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


def evaluate_recursive_multiseries_2(
                                series_train: Union[Dict, pd.DataFrame],
                                series: Union[Dict, pd.DataFrame],
                                model: object, 
                                forecast_horizon: int, 
                                window_stats: List[str], 
                                window_sizes: List[int], 
                                lags: List[int],
                                exog_train: Union[Dict, pd.DataFrame] = None,
                                exog: Union[Dict, pd.DataFrame] = None,
                                transformer_exog: object = None,
                                encoding: str = "onehot",
                                differentiation: int = 1, 
                                refit: Union[bool,int] = 8,
                                fixed_train_size: bool = True,
                                suppress_warnings: bool = False,
                                show_progress: bool = True,
                                n_jobs: Union[str,int] = "auto",
                                verbose: bool = False,
                                metric: str = "mean_absolute_percentage_error"):
    
    warnings.simplefilter('ignore', category=MissingValuesWarning)

    # Fit forecaster
    # ==============================================================================
    # Define the forecaster. 
    window_features = RollingFeatures(stats=window_stats, window_sizes=window_sizes)
    forecaster_recursive = ForecasterRecursiveMultiSeries(
                        regressor        = model,
                        lags             = lags,
                        window_features  = window_features,
                        differentiation  = differentiation,
                        differentiator   = "pct",
                        transformer_exog = transformer_exog,
                        encoding         = encoding,
                    )
    
    forecaster_recursive.fit(series=series_train, 
                            exog=exog_train,
                                suppress_warnings=suppress_warnings)
    

    # Backtesting
    # ==============================================================================
    cv = TimeSeriesFold(
            steps              = forecast_horizon,
            initial_train_size = len(next(iter(series_train.values()))) if isinstance(series_train, dict) else series_train.shape[0], # get first key to obtain length of series
            refit              = refit,
            fixed_train_size   = fixed_train_size,
            differentiation    = differentiation
        )
    

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


# Function to plot predicted intervals
# ======================================================================================
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


def empirical_coverage(y, lower_bound, upper_bound):
    """
    Calculate coverage of a given interval
    """
    return np.mean(np.logical_and(y >= lower_bound, y <= upper_bound))
