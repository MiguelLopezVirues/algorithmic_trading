import sys
sys.path.append("..")
from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
import plotly.subplots as sp
from typing import Tuple, Union, Optional
import yfinance as yf


seed = 8523

def calculate_outcome_n_end_price(backtest_prediction_analysis_df):
    return backtest_prediction_analysis_df.with_columns( # Outcome
                                                    pl.when(pl.col("high") >=  pl.col(f"upper_bound"))
                                                    .then(pl.lit("TP"))
                                                    .when(pl.col("low") <=  pl.col(f"lower_bound"))
                                                    .then(pl.lit("SL"))
                                                    .when((pl.col("forecast_day") == 5))
                                                    .then(pl.lit("close"))
                                                    .otherwise(None).alias("outcome")
                                        ).with_columns( # Endprice
                                                    pl.when(pl.col("outcome") ==  pl.lit("TP"))
                                                    .then(pl.col(f"upper_bound"))
                                                    .when(pl.col("outcome")  ==  pl.lit("SL"))
                                                    .then(pl.col(f"lower_bound"))
                                                    .when((pl.col("outcome") == pl.lit("close")))
                                                    .then(pl.col("close")).otherwise(None).alias("end_price")
                                                    )

def aggregate_outcome_by_trade_id(backtesting_df):
    return backtesting_df.group_by(["trade_id","symbol"]).agg(
                                            pl.col("outcome").drop_nulls().first().alias("outcome"),
                                            pl.col("open").first().alias("open_day_1"),
                                            pl.col(f"upper_bound").mean().alias("mean_upper_bound"),
                                            pl.col(f"lower_bound").mean().alias("mean_lower_bound"),
                                            pl.col("high").max().alias("max_high"),
                                            pl.col("low").min().alias("min_low"),
                                            ((pl.col(f"upper_bound").mean() - pl.col("open").first()) / (pl.col("open").first() - pl.col(f"lower_bound").mean())).alias("mean_RRR"),
                                            ((pl.col(f"upper_bound").mean() - pl.col("open").first()) / pl.col("open").first()).alias("mean_pot_profit_pct"),
                                            pl.col("end_price").drop_nulls().first().alias("first_end_price"),
                                            pl.col("close").last().alias("last_day_close")
                                        ).sort(["trade_id","symbol"])

def calculate_pnl_pct(backtesting_df_aggregated):
    return backtesting_df_aggregated.with_columns((pl.when(pl.col("first_end_price").is_not_null())
                                                                                                  .then(pl.col("first_end_price") - pl.col("open_day_1"))
                                                                                                   .otherwise(pl.col("last_day_close") - pl.col("open_day_1"))).alias("realized_PnL"),
                                                                                                (pl.when(pl.col("first_end_price").is_not_null())
                                                                                                  .then(pl.col("first_end_price") - pl.col("open_day_1"))
                                                                                                   .otherwise(pl.col("last_day_close") - pl.col("open_day_1")) / pl.col("open_day_1") ).alias("realized_PnL_pct"))


def define_risk_rewards(backtesting_pnl_df: pl.DataFrame, rr_ratio_list: List[Union[int, float]]):
    return backtesting_pnl_df.with_columns(
                                    *[pl.when((pl.col("mean_RRR") > rr_ratio)) \
                                    .then(1).otherwise(0) \
                                    .alias(f">trade_{rr_ratio}_ratio") for rr_ratio in rr_ratio_list],
                                    pl.when(pl.col("realized_PnL_pct") > 0).then(1).otherwise(0).alias("win"))



def add_day_trade_identifier_columns(predition_interval_df_symbol: pl.DataFrame) -> pl.DataFrame:
    """
    Adds 'forecast_day' and 'trade_id' columns to the DataFrame.
    
    Args:
        predition_interval_df_symbol (pl.DataFrame): Input DataFrame containing prediction intervals.
        
    Returns:
        pl.DataFrame: DataFrame with added 'forecast_day' and 'trade_id' columns.
    """
    # predition_interval_df_symbol = predition_interval_df_symbol.with_columns(
    #     pl.Series("forecast_day", np.tile(range(1, 6), len(predition_interval_df_symbol) // 5)),
    #     pl.Series("trade_id", [i // 5 for i in range(0, predition_interval_df_symbol.shape[0])])
    # )
    predition_interval_df_symbol = predition_interval_df_symbol.with_columns(
        pl.Series("forecast_day", np.tile(range(1,6), len(predition_interval_df_symbol) // 5)),
        pl.Series("week",[i//5 for i in range(0,predition_interval_df_symbol.shape[0])]))
    return predition_interval_df_symbol



def unpivot_table(preditions_interval_df_wide: pl.DataFrame) -> pl.DataFrame:
    """
    Unpivots the wide-format DataFrame to a long-format DataFrame.
    
    Args:
        preditions_interval_df_wide (pl.DataFrame): Wide-format DataFrame with upper and lower bounds.
        
    Returns:
        pl.DataFrame: Long-format DataFrame with 'symbol', 'upper_bound', and 'lower_bound' columns.
    """

    # Unpivot upper bound
    table_upper_bound = preditions_interval_df_wide.unpivot(
                       index = ["datetime","forecast_day","week"],
                       on = [col for col in preditions_interval_df_wide.columns if "upper" in col],
                       variable_name = "symbol",
                       value_name = "upper_bound"
                       ).with_columns(
                            pl.col("symbol").str.replace("_upper_bound","")
                        )
    
    # Unpivot lower bound
    table_lower_bound = preditions_interval_df_wide.unpivot(
                        index=["datetime"],
                        on=[col for col in preditions_interval_df_wide.columns if "lower" in col],
                        variable_name="symbol",
                        value_name="lower_bound"
                        ).with_columns(
                            pl.col("symbol").str.replace("_lower_bound", "")
                        )

    preditions_interval_df_long = table_upper_bound.join(table_lower_bound, on=["datetime", "symbol"])

    return preditions_interval_df_long
    




def calculate_average_bounds(long_table_with_bounds: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates the average upper and lower bounds for each trade and symbol.
    
    Args:
        long_table_with_bounds (pl.DataFrame): Long-format DataFrame with bounds.
        
    Returns:
        pl.DataFrame: DataFrame with average bounds per trade and symbol.
    """
    long_table_avg_bounds = long_table_with_bounds.group_by(["trade_id", "symbol"]).agg(
        pl.col("datetime").first().alias("forecast_horizon_start"),
        pl.col("upper_bound").mean().alias("mean_upper_bound"),
        pl.col("lower_bound").mean().alias("mean_lower_bound")
    ).sort(["trade_id", "symbol"])
    
    return long_table_avg_bounds

def calculate_necessary_open_per_trade(average_trade_bounds: pl.DataFrame, risk_reward_ratio: float = 1.2, commission_rate: float = 0.002, how: str = "open") -> pl.DataFrame:
    """
    Calculates the required open price for each trade based on risk-reward ratio and commission rate.
    
    Args:
        average_trade_bounds (pl.DataFrame): DataFrame with average bounds per trade.
        risk_reward_ratio (float): Desired risk-reward ratio. Defaults to 1.2.
        commission_rate (float): Commission rate per trade. Defaults to 0.002.
        how (str): Method to apply commission rate. Can be 'open' or 'close'. Defaults to 'open'.
        
    Returns:
        pl.DataFrame: DataFrame with calculated open prices and potential profit percentages.
    """
    if how == "open":
        correction = (1 + commission_rate)  # Add commission
    elif how == "close":
        correction = (1 - commission_rate)  # Subtract commission
        risk_reward_ratio = 1 / risk_reward_ratio
    else:
        raise ValueError("'how' method can only be 'open' or 'close'")
    
    # Calculate required open price and adjusted open price
    opens_for_risk_reward_ratio = average_trade_bounds.with_columns(
        ((pl.col("mean_upper_bound") + pl.col("mean_lower_bound") * risk_reward_ratio) / (risk_reward_ratio + 1)).alias("open_price_for_rrr")
    ).with_columns(
        (pl.col("open_price_for_rrr") / correction).alias("adj_open_price_for_rrr")
    ).with_columns(
        ((((pl.col("mean_upper_bound") + pl.col("mean_lower_bound")) / 2) - pl.col("adj_open_price_for_rrr")) / pl.col("adj_open_price_for_rrr")).alias("mean_pot_profit_pct")
    )
    
    return opens_for_risk_reward_ratio


def create_long_table(preditions_interval_df: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    """
    Converts a Pandas DataFrame to a Polars DataFrame and adds trade identifiers.
    
    Args:
        preditions_interval_df (pd.DataFrame): Input DataFrame in wide format.
        
    Returns:
        pl.DataFrame: Long-format DataFrame with bounds and trade identifiers.
    """
    if isinstance(preditions_interval_df, pd.DataFrame):
        preditions_interval_df = pl.from_pandas(preditions_interval_df, include_index=True).rename({"None":"datetime"})

    table_with_id = add_day_trade_identifier_columns(preditions_interval_df)

    long_table_with_bounds = unpivot_table(table_with_id)

    long_table_with_bounds = long_table_with_bounds.sort(["week","symbol"]).with_columns(pl.Series("trade_id",[i//5 for i in range(0,long_table_with_bounds.shape[0])]))

    return long_table_with_bounds



def trade_evaluation_table(preditions_interval_df: pd.DataFrame, risk_reward_ratio: float = 1.2, commission_rate: float = 0.002, how: str = "open") -> pl.DataFrame:
    """
    Creates a table that allows for quick trade evaluation according to a defined Risk/Reward and comission rate
    by calculating required open prices and potential profits.
    
    Args:
        preditions_interval_df (pd.DataFrame): Input DataFrame with prediction intervals.
        risk_reward_ratio (float): Desired risk-reward ratio. Defaults to 1.2.
        commission_rate (float): Commission rate per trade. Defaults to 0.002.
        how (str): Method to apply commission rate. Can be 'open' or 'close'. Defaults to 'open'.
        
    Returns:
        pl.DataFrame: DataFrame with trade evaluation metrics.
    """
    long_table_with_bounds = create_long_table(preditions_interval_df)
    avg_bounds_per_trade = calculate_average_bounds(long_table_with_bounds)
    trade_required_open_prices_n_profit = calculate_necessary_open_per_trade(avg_bounds_per_trade, risk_reward_ratio=risk_reward_ratio, commission_rate=commission_rate, how=how)
    return trade_required_open_prices_n_profit



def create_bounds_dict(long_table_predictions: pl.DataFrame) -> Dict[str, Dict]:
    """
    Creates a dictionary of bounds for each symbol.
    
    Args:
        long_table_predictions (pl.DataFrame): Long-format DataFrame with bounds.
        
    Returns:
        Dict[str, Dict]: Dictionary where keys are symbols and values are bounds.
    """
    return {
        symbol: long_table_predictions.filter(pl.col("symbol") == symbol)
        .select(pl.exclude(["symbol", "trade_id"]))
        .rename({"datetime": "forecast_horizon_start"})
        .with_columns(pl.col("forecast_horizon_start").dt.strftime("%Y-%m-%d"))
        .to_pandas()
        .set_index("forecast_horizon_start")
        .to_dict(orient="index")
        for symbol in long_table_predictions["symbol"].unique().to_list()
    }



def create_trade_mongodb_documents(predictions: pd.DataFrame, target_risk_reward: float, comission_rate: float) -> List[Dict]:
    """
    Creates MongoDB documents for trade evaluations to be upload to the database .
    
    Args:
        predictions (pd.DataFrame): Input DataFrame with prediction intervals.
        target_risk_reward (float): Target risk-reward ratio.
        comission_rate (float): Commission rate per trade.
        
    Returns:
        List[Dict]: List of dictionaries representing MongoDB documents.
    """
    predict_evaluation_table = trade_evaluation_table(predictions)
    predict_evaluation_table = predict_evaluation_table.with_columns(
        pl.lit(datetime.today().strftime("%Y-%m-%d")).alias("forecast_creation"),
        pl.lit(target_risk_reward).alias("target_risk_reward"),
        pl.lit(comission_rate).alias("comission_rate")
    )
    
    long_table_predictions = create_long_table(predictions)
    bounds_dict = create_bounds_dict(long_table_predictions)
    
    predict_evaluation_table = predict_evaluation_table.select(pl.exclude("trade_id")).with_columns(
        pl.col("symbol").map_elements(lambda s: bounds_dict.get(s, None)).alias("bounds")
    )
    
    return predict_evaluation_table.to_pandas().to_dict(orient="records")


def create_backtesting_df(backtesting_predictions: pl.DataFrame, ohlcv_data: pl.DataFrame, symbol_column: str = "symbol", datetime_column: str = "datetime"):
    predictions_long_format = create_long_table(backtesting_predictions)

    merged_with_ohlcv = predictions_long_format.join(ohlcv_data, on=["symbol","datetime"], how="left").sort([symbol_column,datetime_column])

    return merged_with_ohlcv

def calculate_metrics_for_backtesting(backtesting_predictions_with_ohlcv: pl.DataFrame):
    predictions_with_metrics = backtesting_predictions_with_ohlcv.with_columns( 
                                                                                mean_upper_bound=pl.col("upper_bound").mean().over("trade_id"),
                                                                                mean_lower_bound=pl.col("lower_bound").mean().over("trade_id")
                                                                ).with_columns(
                                                                                first_open=pl.col("open").first().over("trade_id")
                                                                ).with_columns(
                                                                                rr_ratio=(pl.col("mean_upper_bound") - pl.col("first_open")) 
                                                                                / (pl.col("first_open") - pl.col("mean_lower_bound"))
                                                                               )
    return predictions_with_metrics




## BACKTESTING

def backtest_strategy(  backtest_predictions_df: pl.DataFrame, 
                        ohlcv_data: pl.DataFrame,
                        top_operations: int = 10,
                        min_rr_ratio: int = 2,
                        wallet: int = 100000,
                        fraction_per_investment: float = 0.1,
                        commission_rate: float = 0.000,
                        invest_max: bool = False,
                        verbose: bool = False
                        ) -> Tuple[pl.DataFrame,pl.DataFrame]:
    
    initial_wallet = wallet
    # Create necessary metrics for backtesting
    backtesting_data_with_ohlcv = create_backtesting_df(backtest_predictions_df, ohlcv_data)
    df_metrics = calculate_metrics_for_backtesting(backtesting_data_with_ohlcv)
    
    # Instantiate records
    dfs_weeks_list = []
    weekly_cash_flow = {
        "Date": [],
        "Week": [],
        "Wallet": [],
        "Investment_costs": [],
        "PnL": [],
        "Q_symbols_traded": [],
        "Commission_rate": []
    }

    if top_operations > (1 / fraction_per_investment):
        raise ValueError("Weekly operations cannot surpass 1 / fraction_per_investment")


    for week in df_metrics["week"].unique().to_list():
        # Select top selected symbols
        top_symbols = df_metrics.filter((pl.col("week") == week) 
                                        & ((pl.col("rr_ratio") >= min_rr_ratio))
                                        ).group_by("symbol").agg(pl.col("rr_ratio").first()
                                                                ).sort("rr_ratio", descending=True)[:top_operations]
        
        # If there are no available symbols, continue
        if len(top_symbols) == 0:
            continue
        
        if invest_max:
            fraction_per_investment = 1 / max(2, len(top_symbols))

        # Calculate investment and investment cost
        investment = wallet * fraction_per_investment
        week_cost = - len(top_symbols) * investment * commission_rate

        # Calculate selected trades results
        df_iter_symbols = df_metrics.filter( # Filter by current week and selected symbols
                                            (pl.col("week") == week), pl.col("symbol").is_in(top_symbols["symbol"].to_list()))
        
        df_trades_week = df_iter_symbols.with_columns(      # Calculate TP/SL or close outcome
                                                            pl.when(pl.col("forecast_day") == 1).then(1).otherwise(0).alias("open_trade"),
                                                            pl.when(pl.col("low") < pl.col("lower_bound")).then(pl.lit("SL")
                                                            ).when(pl.col("high") > pl.col("upper_bound")).then(pl.lit("TP")
                                                            ).when(pl.col("forecast_day") == 5).then(pl.lit("close")).alias("outcome")
                            
                                        ).with_columns(     # Calculate close price for outcome
                                                            pl.when(pl.col("outcome")=="SL").then(pl.col("lower_bound")
                                                            ).when(pl.col("outcome")=="TP").then(pl.col("upper_bound")
                                                            ).when(pl.col("outcome")=="close").then(pl.col("close")).alias("final_price")

                                        ).sort(["symbol","datetime"]
                                        ).group_by("symbol").agg(   # Calculate % PnL per trade
                                                                    pl.col("datetime").first(),
                                                                    pl.col("week").first(),
                                                                    ((pl.col("final_price").drop_nulls().first() - pl.col("open").first()) / pl.col("open").first()).alias("pct_profit")
                                                                    
                                                                    # Calculate PnL based on investment
                                                                    ).sort("symbol").with_columns((pl.col("pct_profit")*investment).alias("PnL"))
        
        # Calculate weekly results
        week_pnl = df_trades_week["PnL"].sum()
        wallet += week_pnl - len(top_symbols) * investment * commission_rate

        # Record results
        dfs_weeks_list.append(df_trades_week)
        weekly_cash_flow["Date"].append(df_trades_week["datetime"][0])
        weekly_cash_flow["Week"].append(week)
        weekly_cash_flow["Wallet"].append(wallet)
        weekly_cash_flow["PnL"].append(week_pnl)
        weekly_cash_flow["Q_symbols_traded"].append(len(top_symbols))
        weekly_cash_flow["Investment_costs"].append(week_cost)
        weekly_cash_flow["Commission_rate"].append(commission_rate)


        if verbose:
            print("===============")
            print(f"Week {week}")
            print("Q symbols traded:", len(top_symbols))
            print("Total investment:", len(top_symbols) * investment)
            print("Comission rate expenses:", len(top_symbols) * investment * commission_rate)
            print("Week_pnl:", week_pnl)
            print("Wallet:", wallet)
    

    weekly_trades = pl.concat(dfs_weeks_list, how="vertical")
    weekly_trades = weekly_trades.with_columns((pl.col("PnL").cum_sum() + initial_wallet).alias("Wallet"))
    weekly_cash_flow = pl.DataFrame(weekly_cash_flow)

    return weekly_trades, weekly_cash_flow


## COMPARISON METRICS

def compute_performance_metrics(df: pl.DataFrame) -> dict:
    df = df.with_columns(
        (df["Wallet"].pct_change()).alias("Weekly_Return")
    ).drop_nulls()
    
    start_equity = df["Wallet"].first()
    end_equity = df["Wallet"].last()
    duration_weeks = df.height
    
    # Annualized Return (CAGR)
    years = duration_weeks / 52
    return_pct = (end_equity / start_equity - 1) * 100
    return_ann = ((end_equity / start_equity) ** (1 / years) - 1) * 100
    
    # Volatility (Annualized)
    volatility_ann = df["Weekly_Return"].std() * np.sqrt(52) * 100
    
    # Sharpe Ratio (Risk-free rate = 0)
    sharpe_ratio = return_ann / volatility_ann if volatility_ann != 0 else None
    
    # Sortino Ratio (Downside Volatility)
    negative_returns = df.filter(df["Weekly_Return"] < 0)["Weekly_Return"]
    downside_vol = negative_returns.std() * np.sqrt(52) * 100 if negative_returns.len() > 0 else 0
    sortino_ratio = return_ann / downside_vol if downside_vol != 0 else None
    
    return {
        "Equity Final [$]": end_equity,
        "Return [%]": return_pct,
        "Return (Ann.) [%]": return_ann,
        "Volatility (Ann.) [%]": volatility_ann,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
    }

def compute_drawdown_metrics(df: pl.DataFrame) -> dict:
    df = df.with_columns(
        (df["Wallet"].cum_max()).alias("Peak")
    )
    df = df.with_columns(
        ((df["Wallet"] - df["Peak"]) / df["Peak"]).alias("Drawdown")
    )
    
    max_drawdown = df["Drawdown"].min() * 100
    avg_drawdown = df["Drawdown"].mean() * 100
    
    # Drawdown Durations
    df = df.with_columns((df["Drawdown"] < 0).cast(pl.Int32).alias("Drawdown_Flag"))
    df = df.with_columns((df["Drawdown_Flag"].diff() == 1).cum_sum().alias("Drawdown_ID"))
    drawdown_durations = df.group_by("Drawdown_ID").agg(pl.len().alias("Duration"))["Duration"]
    
    max_drawdown_duration = drawdown_durations.max()
    avg_drawdown_duration = drawdown_durations.mean()
    
    return {
        "Max. Drawdown [%]": max_drawdown,
        "Avg. Drawdown [%]": avg_drawdown,
        "Max. Drawdown Duration [weeks]": max_drawdown_duration,
        "Avg. Drawdown Duration [weeks]": avg_drawdown_duration,
    }

def compute_trade_metrics(df: pl.DataFrame) -> dict:
    df = df.with_columns(
        (df["PnL"] / df["Wallet"].shift(1)).alias("Trade_Return")
    ).drop_nulls()
    
    num_trades = df["Q_symbols_traded"].sum()
    win_rate = (df.filter(df["Trade_Return"] > 0).height / num_trades) * 100 if num_trades else 0
    best_trade = df["Trade_Return"].max() * 100
    worst_trade = df["Trade_Return"].min() * 100
    avg_trade = df["Trade_Return"].mean() * 100
    
    # Amount earned in wins vs amount lost in drawdowns
    profit_factor = df.filter(df["Trade_Return"] > 0)["Trade_Return"].sum() / abs(df.filter(df["Trade_Return"] < 0)["Trade_Return"].sum()) if df.filter(df["Trade_Return"] < 0).height > 0 else None

    # Mean expected return per trade
    expectancy = (df["Trade_Return"].mean() * 100)
    
    return {
        "# Trades": num_trades,
        "Win Rate [%]": win_rate,
        "Best Trade [%]": best_trade,
        "Worst Trade [%]": worst_trade,
        "Avg. Trade [%]": avg_trade,
        "Profit Factor": profit_factor,
        "Expectancy [%]": expectancy,
    }

def compute_all_metrics(df: pl.DataFrame) -> dict:
    return {
        **compute_performance_metrics(df),
        **compute_drawdown_metrics(df),
        **compute_trade_metrics(df),
    }


def buy_hold_stock_performance_metrics(stock_ohlcv_data: Optional[pl.DataFrame] = None, ticker: Optional[str]=None, start: Optional[str] = None, end: Optional[str] = None):
    if ticker:
       stock_ohlcv_data = pl.from_pandas(yf.download(ticker, start=start, end=end, 
                                                     progress=False, multi_level_index=False), include_index=True).select(pl.all().name.to_lowercase())
       
    else:
        # Ensure the dataframe has the necessary columns: 'date', 'open', 'high', 'low', 'close', 'volume'
        assert all(col in stock_ohlcv_data.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume']), \
            "DataFrame must contain 'date', 'open', 'high', 'low', 'close', and 'volume' columns."

    # Calculate daily return
    stock_ohlcv_data = stock_ohlcv_data.with_columns(
        ((pl.col('close').shift(-1) - pl.col('close')) / pl.col('close').shift(-1) * 100).alias('daily_return'))
    
    # Calculate cumulative return
    stock_ohlcv_data = stock_ohlcv_data.with_columns(
        (pl.col('daily_return') / 100 + 1).cum_prod().alias('cumulative_return')
    )
    stock_ohlcv_data = stock_ohlcv_data.drop_nulls()

    # Calculate volatility (standard deviation of daily returns)
    volatility = stock_ohlcv_data['daily_return'].std()

    # Calculate Sharpe Ratio assuming a risk-free rate of 0% (can be adjusted as needed)
    sharpe_ratio = stock_ohlcv_data['daily_return'].mean() / volatility if volatility != 0 else np.nan
    
    # Calculate Sortino Ratio: downside deviation (negative returns) instead of total volatility
    downside_returns = stock_ohlcv_data.filter(pl.col('daily_return') < 0)
    downside_volatility = downside_returns['daily_return'].std()
    sortino_ratio = stock_ohlcv_data['daily_return'].mean() / downside_volatility if downside_volatility != 0 else np.nan
    
    # Calculate Maximum Drawdown
    stock_ohlcv_data = stock_ohlcv_data.with_columns(
        pl.col('cumulative_return').cum_max().alias('max_cumulative_return')
    )
    stock_ohlcv_data = stock_ohlcv_data.with_columns(
        (pl.col('cumulative_return') - pl.col('max_cumulative_return')).alias('drawdown')
    )
    max_drawdown = stock_ohlcv_data['drawdown'].min()

    # Calculate Max Drawdown Duration
    drawdown_duration = stock_ohlcv_data.filter(pl.col('drawdown') < 0).select(pl.col('date')).shape[0]

    return stock_ohlcv_data, {
        'Return [%]': (stock_ohlcv_data["cumulative_return"][-1] - 1) * 100,
        'Volatility': volatility * 100,
        'Sharpe ratio': sharpe_ratio,
        'Sortino ratio': sortino_ratio,
        'Max drawdown [%]': max_drawdown * 100,
        'Max drawdown duration [weeks]': int(drawdown_duration / 5)
    }


def plot_strategy_vs_ticker(strategy_df: pl.DataFrame, ticker: str, metrics: dict = None):
    
    ticker_df = pl.from_pandas(yf.download(ticker, start=strategy_df["Date"][0], end=strategy_df["Date"][-1], progress=False, multi_level_index=False), include_index=True)

    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.3, 0.7],
                           subplot_titles=("Strategy Performance", f"{ticker} Candlestick Chart"))
    
    # Plot strategy equity curve
    fig.add_trace(go.Scatter(x=strategy_df["Date"], y=strategy_df["Wallet"], mode='lines',
                             name='Strategy Equity', line=dict(color='blue')), row=1, col=1)
    
    # Add performance metrics as annotations
    if metrics:
        annotations = []
        for i, (key, value) in enumerate(metrics.items()):

            annotations.append(dict(
                xref='paper', 
                yref='paper', 
                x=0.01, 
                y=1.05 - i * 0.065,
                text=f"{key}: {value:.2f}" if value else f"{key}: None", 
                showarrow=False, 
                font=dict(size=12),
                bgcolor='rgba(255, 255, 255, 0.8)',  # Black background with 10% opacity
                borderpad=4  # Optional: Adds padding around the text
            ))

        fig.update_layout(annotations=annotations)
    
    # Plot ticker candlestick chart
    fig.add_trace(go.Candlestick(x=ticker_df["Date"], open=ticker_df["Open"], high=ticker_df["High"],
                                 low=ticker_df["Low"], close=ticker_df["Close"], name=ticker), row=2, col=1)
    
    fig.update_layout(title_text=f"Strategy vs. Buy-Hold of {ticker} Comparison", xaxis_rangeslider_visible=False, height=800)
    fig.show()