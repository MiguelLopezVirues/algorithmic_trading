import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def ticker_candlestick_plot(ticker_df: pd.DataFrame, ticker_name: str ="Stock", height_px: int=600, slider_visible: bool=False):
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