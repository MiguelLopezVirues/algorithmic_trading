import pandas as pd
import numpy as np 
import polars as pl

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import ccf



def ticker_candlestick_plot(ticker_df: pd.DataFrame, 
                            ticker_name: str ="Stock", 
                            height_px: int=600, 
                            slider_visible: bool=False):
    """
    Generates and displays a candlestick chart for a given stock ticker using Plotly.

    Args:
        ticker_df (pd.DataFrame): DataFrame containing stock data with columns ['open', 'high', 'low', 'close'] 
                                  and a DateTime index.
        ticker_name (str, optional): Name of the stock to display in the chart title. Defaults to "Stock".
        height_px (int, optional): Height of the chart in pixels. Defaults to 600.
        slider_visible (bool, optional): Whether to show the range slider for the x-axis. Defaults to False.

    Returns:
        None: Displays the generated candlestick chart.
    """
    ticker_df.columns = [col.lower() for col in ticker_df.columns]
    # define figure
    fig = go.Figure(data=[go.Candlestick(x=ticker_df.index,
                    open=ticker_df["open"],
                    high=ticker_df["high"],
                    low=ticker_df["low"],
                    close=ticker_df["close"])
                ])
    
    min_date_str = min(ticker_df.index).strftime('%Y-%m-%d')
    max_date_str = max(ticker_df.index).strftime('%Y-%m-%d')

    fig.update_layout(
        title=f"{ticker_name}'s daily candlestick chart from {min_date_str} to {max_date_str}",
        title_x=0.5,  # Set title x-position to center
        xaxis_rangeslider_visible=slider_visible,
        height=height_px
        )

    fig.show()


def plot_acf_pacf(data, lags=36, method="ywmle"):
    """
    Grafica las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF).
    
    Parameters:
    -----------
    lags : int
        Número de lags a graficar.
    """
    plot_acf(data=data, lags=lags, method=method)
    plot_pacf(data=data, lags=lags, method=method)


def make_plot_acf(data, lags=36, method="ywmle"):
    plt.figure(figsize=(12, 10))
    plot_acf(data.drop_nulls(), lags=lags, method=method)
    plt.title("Función de Autocorrelación (ACF)")
    plt.grid()
    plt.show()

def make_plot_pacf(data, lags=36, method="ywmle"):
    plt.figure(figsize=(12, 10))
    plot_pacf(data.drop_nulls(), lags=lags, method=method)
    plt.title("Función de Autocorrelación Parcial (PACF)")
    plt.grid()
    plt.show()



def subplots_timeseries_lags(timeseries_df: pl.DataFrame, ticker_column: str) -> None:
    df = timeseries_df

    # Define the lags
    trading_day_lags = {
        '1d': 1,
        # '3d': 3,
        '7d': 5,
        # '30d': 22,
        # '90d': 66,
        '365d': 252
    }

    # Create subplots
    fig = make_subplots(rows=len(trading_day_lags) + 1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Plot the original time datetime
    fig.add_trace(go.Scatter(x=df["datetime"], y=df[ticker_column], name='Original'), row=1, col=1)

    # Plot the lagged time series in the dictionary
    for i, (lag_name, lag_value) in enumerate(trading_day_lags.items(), start=2):

        lagged_series = df.with_columns(
                                        (pl.col(ticker_column).pct_change(lag_value) * 100).alias("lagged")  # Calculate percentage change
                                    ).filter(pl.col("lagged").is_not_null())  # Drop nulls

        fig.add_trace(go.Scatter(x=lagged_series["datetime"], y=lagged_series["lagged"], name=f'Growth rate {lag_name}'), row=i, col=1)


    # Add a date filter slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),  # Set rangeslider to False to remove it
            type="date"
        )
    )

    # Update layout for better visualization
    fig.update_layout(height=800, title_text=f"{ticker_column} Time Series with Lags", showlegend=True)

    fig.show()


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
    # Align on the common index to avoid misalignment
    x = x.dropna()
    y = y.dropna()
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx]
    y = y.loc[common_idx]

    # onvert to numpy arrays for numpy ccf
    x_arr = x.values
    y_arr = y.values

    # ccf, correlation for lags = 0..(len(x)-1)
    ccf_xy = ccf(x_arr, y_arr) 
    ccf_yx = ccf(y_arr, x_arr)

    # Truncate to max_lag + 1 terms from each
    ccf_xy_pos_lags = ccf_xy[: max_lag + 1]  
    ccf_yx_pos_lags = ccf_yx[: max_lag + 1]  

  
    neg_lags = np.arange(-max_lag, 0) # To form negative lags, invert the range
    pos_lags = np.arange(0, max_lag + 1)

    # slice to skip the zero-lag, then reverse
    ccf_neg = ccf_yx_pos_lags[1:][::-1]  # Lags: (max_lag..1) reversed => -1..-max_lag
    ccf_pos = ccf_xy_pos_lags            # Lags: 0..+max_lag

    # Combine symmetrical range
    ccf_full = np.concatenate([ccf_neg, ccf_pos])
    lags_full = np.concatenate([neg_lags, pos_lags])

    # 95% confidence intervals ~ ±2/sqrt(N)
    N = len(x_arr)
    conf = 2.0 / np.sqrt(N)

    # Plot
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
