import sys
sys.path.append("..")
from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List


seed = 8523

def add_day_trade_identifier_columns(predition_interval_df_symbol: pl.DataFrame) -> pl.DataFrame:
    """
    Adds 'forecast_day' and 'trade_id' columns to the DataFrame.
    
    Args:
        predition_interval_df_symbol (pl.DataFrame): Input DataFrame containing prediction intervals.
        
    Returns:
        pl.DataFrame: DataFrame with added 'forecast_day' and 'trade_id' columns.
    """
    predition_interval_df_symbol = predition_interval_df_symbol.with_columns(
        pl.Series("forecast_day", np.tile(range(1, 6), len(predition_interval_df_symbol) // 5)),
        pl.Series("trade_id", [i // 5 for i in range(0, predition_interval_df_symbol.shape[0])])
    )
    return predition_interval_df_symbol



def unpivot_table(preditions_interval_df_wide: pl.DataFrame) -> pl.DataFrame:
    """
    Unpivots the wide-format DataFrame to a long-format DataFrame.
    
    Args:
        preditions_interval_df_wide (pl.DataFrame): Wide-format DataFrame with upper and lower bounds.
        
    Returns:
        pl.DataFrame: Long-format DataFrame with 'symbol', 'upper_bound', and 'lower_bound' columns.
    """

    # Unpivot upper bounds
    table_upper_bound = preditions_interval_df_wide.unpivot(
        index=["datetime", "forecast_day", "trade_id"],
        on=[col for col in preditions_interval_df_wide.columns if "upper" in col],
        variable_name="symbol",
        value_name="upper_bound"
    ).with_columns(
        pl.col("symbol").str.replace("_upper_bound", "")
    )
    
    # Unpivot lower bounds
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


def create_long_table(preditions_interval_df: pd.DataFrame) -> pl.DataFrame:
    """
    Converts a Pandas DataFrame to a Polars DataFrame and adds trade identifiers.
    
    Args:
        preditions_interval_df (pd.DataFrame): Input DataFrame in wide format.
        
    Returns:
        pl.DataFrame: Long-format DataFrame with bounds and trade identifiers.
    """
    preditions_interval_df = pl.from_pandas(preditions_interval_df, include_index=True).rename({"None": "datetime"})
    table_with_id = add_day_trade_identifier_columns(preditions_interval_df)
    long_table_with_bounds = unpivot_table(table_with_id)
    return long_table_with_bounds



def trade_evaluation_table(preditions_interval_df: pd.DataFrame, risk_reward_ratio: float = 1.2, commission_rate: float = 0.002, how: str = "open") -> pl.DataFrame:
    """
    Evaluates trades by calculating required open prices and potential profits.
    
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
    Creates MongoDB documents for trade evaluations.
    
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