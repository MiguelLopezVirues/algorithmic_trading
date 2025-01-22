import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
import re
import talib
from typing import List, Optional
import pandas_market_calendars as mcal
import warnings
from tqdm  import tqdm

from .file_handling import FileHandler


# TO-DO:
# [] add error-handling

# class FileHandler:
#     def read_csv_file(self, file_path: str) -> pd.DataFrame:
#         """
#         Reads a CSV file and parses its index as a datetime.
        
#         Args:
#             file_path (str): Path to the CSV file.
        
#         Returns:
#             pd.DataFrame: The loaded DataFrame with a datetime index.
#         """
#         df = pd.read_csv(file_path, index_col=0)
#         df.index = pd.to_datetime(df.index, utc=True, format="%Y-%m-%d")
#         return df   
    
#     def list_all_files(self, directory: str):
#         """
#         Lists all nested files in a directory.
        
#         Args:
#             directory (str): Path to a directory.
        
#         Returns:
#             List: List with all files, nested or not, inside the directory.
#         """
#         return [file for file in Path(directory).rglob('*') if file.is_file()]

#     def save_dataframe_csv_file(self, df: pd.DataFrame, save_path: str) -> None:
#         """
#         Saves a DataFrame to a CSV file, creating necessary directories.
        
#         Args:
#             df (pd.DataFrame): The DataFrame to save.
#             save_path (str): Path to save the CSV file.
#         """
#         save_path_dir = Path(save_path).parent
#         save_path_dir.mkdir(parents=True, exist_ok=True)

#         df.to_csv(save_path)

#     def read_transform_save(
#         self,
#         transform_function: Callable[[pd.DataFrame], pd.DataFrame], 
#         read_path: str, 
#         save_path: Optional[str] = None
#     ) -> pd.DataFrame:
#         """
#         Reads a CSV, applies a transformation function, and saves the result.

#         Args:
#             transform_function (Callable): Function to transform the DataFrame.
#             read_path (str): Path to the input CSV file.
#             save_path (Optional[str]): Path to save the transformed CSV file. If None, a default path is generated.

#         Returns:
#             pd.DataFrame: Transformed DataFrame.
#         """

#         # Read
#         df = self.read_csv_file(read_path)

#         # Transform
#         df = transform_function(df)

#         # Save
#         if not save_path:
#             save_path = re.sub(r"extracted","transformed",read_path)
        
#         self.save_dataframe_csv_file(df, save_path)

#         return df


    
class TechnicalIndicators(FileHandler):
    def talib_get_momentum_indicators_for_one_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        # ADX - Average Directional Movement Index
        talib_momentum_adx = talib.ADX(df.high.values, df.low.values, df.close.values, timeperiod=14)
        # ADXR - Average Directional Movement Index Rating
        talib_momentum_adxr = talib.ADXR(df.high.values, df.low.values, df.close.values, timeperiod=14 )
        # APO - Absolute Price Oscillator
        talib_momentum_apo = talib.APO(df.close.values, fastperiod=12, slowperiod=26, matype=0 )
        # AROON - Aroon
        talib_momentum_aroon = talib.AROON(df.high.values, df.low.values, timeperiod=14 )
        # talib_momentum_aroon[0].size
        # talib_momentum_aroon[1].size
        # AROONOSC - Aroon Oscillator
        talib_momentum_aroonosc = talib.AROONOSC(df.high.values, df.low.values, timeperiod=14)
        # BOP - Balance of Power
        # https://school.stockcharts.com/doku.php?id=technical_indicators:balance_of_power
        #calculate open prices as shifted closed prices from the prev day
        # open = df.Last.shift(1)
        talib_momentum_bop = talib.BOP(df.open.values, df.high.values, df.low.values, df.close.values)
        # CCI - Commodity Channel Index
        talib_momentum_cci = talib.CCI(df.high.values, df.low.values, df.close.values, timeperiod=14)
        # CMO - Chande Momentum Oscillator
        talib_momentum_cmo = talib.CMO(df.close.values, timeperiod=14)
        # DX - Directional Movement Index
        talib_momentum_dx = talib.DX(df.high.values, df.low.values, df.close.values, timeperiod=14)
        # MACD - Moving Average Convergence/Divergence
        talib_momentum_macd, talib_momentum_macdsignal, talib_momentum_macdhist = talib.MACD(df.close.values, fastperiod=12, \
                                                                                slowperiod=26, signalperiod=9)
        # MACDEXT - MACD with controllable MA type
        talib_momentum_macd_ext, talib_momentum_macdsignal_ext, talib_momentum_macdhist_ext = talib.MACDEXT(df.close.values, \
                                                                                            fastperiod=12, \
                                                                                            fastmatype=0, \
                                                                                            slowperiod=26, \
                                                                                            slowmatype=0, \
                                                                                            signalperiod=9, \
                                                                                            signalmatype=0)
        # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        talib_momentum_macd_fix, talib_momentum_macdsignal_fix, talib_momentum_macdhist_fix = talib.MACDFIX(df.close.values, \
                                                                                                signalperiod=9)
        # MFI - Money Flow Index
        talib_momentum_mfi = talib.MFI(df.high.values, df.low.values, df.close.values, df.volume.values, timeperiod=14)
        # MINUS_DI - Minus Directional Indicator
        talib_momentum_minus_di = talib.MINUS_DM(df.high.values, df.low.values, timeperiod=14)
        # MOM - Momentum
        talib_momentum_mom = talib.MOM(df.close.values, timeperiod=10)
        # PLUS_DI - Plus Directional Indicator
        talib_momentum_plus_di = talib.PLUS_DI(df.high.values, df.low.values, df.close.values, timeperiod=14)
        # PLUS_DM - Plus Directional Movement
        talib_momentum_plus_dm = talib.PLUS_DM(df.high.values, df.low.values, timeperiod=14)
        # PPO - Percentage Price Oscillator
        talib_momentum_ppo = talib.PPO(df.close.values, fastperiod=12, slowperiod=26, matype=0)
        # ROC - Rate of change : ((price/prevPrice)-1)*100
        talib_momentum_roc = talib.ROC(df.close.values, timeperiod=10)
        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        talib_momentum_rocp = talib.ROCP(df.close.values, timeperiod=10)
        # ROCR - Rate of change ratio: (price/prevPrice)
        talib_momentum_rocr = talib.ROCR(df.close.values, timeperiod=10)
        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        talib_momentum_rocr100 = talib.ROCR100(df.close.values, timeperiod=10)
        # RSI - Relative Strength Index
        talib_momentum_rsi = talib.RSI(df.close.values, timeperiod=14)
        # STOCH - Stochastic
        talib_momentum_slowk, talib_momentum_slowd = talib.STOCH(df.high.values, df.low.values, df.close.values, \
                                                    fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        # STOCHF - Stochastic Fast
        talib_momentum_fastk, talib_momentum_fastd = talib.STOCHF(df.high.values, df.low.values, df.close.values, \
                                                    fastk_period=5, fastd_period=3, fastd_matype=0)
        # STOCHRSI - Stochastic Relative Strength Index
        talib_momentum_fastk_rsi, talib_momentum_fastd_rsi = talib.STOCHRSI(df.close.values, timeperiod=14, \
                                                                fastk_period=5, fastd_period=3, fastd_matype=0)
        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        talib_momentum_trix = talib.TRIX(df.close.values, timeperiod=30)
        # ULTOSC - Ultimate Oscillator
        talib_momentum_ultosc = talib.ULTOSC(df.high.values, df.low.values, df.close.values, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        # WILLR - Williams' %R
        talib_momentum_willr = talib.WILLR(df.high.values, df.low.values, df.close.values, timeperiod=14)

        momentum_df =   pd.DataFrame(
            {
                # assume here multi-index <dateTime, ticker>
                # 'datetime': df.index.get_level_values(0),
                # 'ticker': df.index.get_level_values(1) ,

                # old way with separate columns
                'date': df.date.values,
                'symbol': df.symbol,

                'adx': talib_momentum_adx,
                'adxr': talib_momentum_adxr,
                'apo': talib_momentum_apo,
                'aroon_1': talib_momentum_aroon[0] ,
                'aroon_2': talib_momentum_aroon[1],
                'aroonosc': talib_momentum_aroonosc,
                'bop': talib_momentum_bop,
                'cci': talib_momentum_cci,
                'cmo': talib_momentum_cmo,
                'dx': talib_momentum_dx,
                'macd': talib_momentum_macd,
                'macdsignal': talib_momentum_macdsignal,
                'macdhist': talib_momentum_macdhist,
                'macd_ext': talib_momentum_macd_ext,
                'macdsignal_ext': talib_momentum_macdsignal_ext,
                'macdhist_ext': talib_momentum_macdhist_ext,
                'macd_fix': talib_momentum_macd_fix,
                'macdsignal_fix': talib_momentum_macdsignal_fix,
                'macdhist_fix': talib_momentum_macdhist_fix,
                'mfi': talib_momentum_mfi,
                'minus_di': talib_momentum_minus_di,
                'mom': talib_momentum_mom,
                'plus_di': talib_momentum_plus_di,
                'dm': talib_momentum_plus_dm,
                'ppo': talib_momentum_ppo,
                'roc': talib_momentum_roc,
                'rocp': talib_momentum_rocp,
                'rocr': talib_momentum_rocr,
                'rocr100': talib_momentum_rocr100,
                'rsi': talib_momentum_rsi,
                'slowk': talib_momentum_slowk,
                'slowd': talib_momentum_slowd,
                'fastk': talib_momentum_fastk,
                'fastd': talib_momentum_fastd,
                'fastk_rsi': talib_momentum_fastk_rsi,
                'fastd_rsi': talib_momentum_fastd_rsi,
                'trix': talib_momentum_trix,
                'ultosc': talib_momentum_ultosc,
                'willr': talib_momentum_willr,
            }
        )
        return momentum_df
    
    def talib_get_volume_volatility_cycle_price_indicators(self,df: pd.DataFrame) -> pd.DataFrame:
        # TA-Lib volume indicators
        # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/volume_indicators.md
        # AD - Chaikin A/D Line
        talib_ad = talib.AD(
            df.high.values, df.low.values, df.close.values, df.volume.values)
        # ADOSC - Chaikin A/D Oscillator
        talib_adosc = talib.ADOSC(
            df.high.values, df.low.values, df.close.values, df.volume.values, fastperiod=3, slowperiod=10)
        # OBV - On Balance volume
        talib_obv = talib.OBV(
            df.close.values, df.volume.values)

        # TA-Lib Volatility indicators
        # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/volatility_indicators.md
        # ATR - Average True Range
        talib_atr = talib.ATR(
            df.high.values, df.low.values, df.close.values, timeperiod=14)
        # NATR - Normalized Average True Range
        talib_natr = talib.NATR(
            df.high.values, df.low.values, df.close.values, timeperiod=14)
        # OBV - On Balance volume
        talib_obv = talib.OBV(
            df.close.values, df.volume.values)

        # TA-Lib Cycle Indicators
        # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/cycle_indicators.md
        # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
        talib_ht_dcperiod = talib.HT_DCPERIOD(df.close.values)
        # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
        talib_ht_dcphase = talib.HT_DCPHASE(df.close.values)
        # HT_PHASOR - Hilbert Transform - Phasor Components
        talib_ht_phasor_inphase, talib_ht_phasor_quadrature = talib.HT_PHASOR(
            df.close.values)
        # HT_SINE - Hilbert Transform - SineWave
        talib_ht_sine_sine, talib_ht_sine_leadsine = talib.HT_SINE(
            df.close.values)
        # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
        talib_ht_trendmode = talib.HT_TRENDMODE(df.close.values)

        # TA-Lib Price Transform Functions
        # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/price_transform.md
        # AVGPRICE - Average Price
        talib_avgprice = talib.AVGPRICE(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # MEDPRICE - Median Price
        talib_medprice = talib.MEDPRICE(df.high.values, df.low.values)
        # TYPPRICE - Typical Price
        talib_typprice = talib.TYPPRICE(
            df.high.values, df.low.values, df.close.values)
        # WCLPRICE - Weighted close Price
        talib_wclprice = talib.WCLPRICE(
            df.high.values, df.low.values, df.close.values)

        volume_volatility_cycle_price_df = pd.DataFrame(
            {'date': df.date.values,
                'symbol': df.symbol,
                # TA-Lib volume indicators
                'ad': talib_ad,
                'adosc': talib_adosc,
                'obv': talib_obv,
                # TA-Lib Volatility indicators
                'atr': talib_atr,
                'natr': talib_natr,
                'obv': talib_obv,
                # TA-Lib Cycle Indicators
                'ht_dcperiod': talib_ht_dcperiod,
                'ht_dcphase': talib_ht_dcphase,
                'ht_phasor_inphase': talib_ht_phasor_inphase,
                'ht_phasor_quadrature': talib_ht_phasor_quadrature,
                'ht_sine_sine': talib_ht_sine_sine,
                'ht_sine_leadsine': talib_ht_sine_leadsine,
                'ht_trendmod': talib_ht_trendmode,
                # TA-Lib Price Transform Functions
                'avgprice': talib_avgprice,
                'medprice': talib_medprice,
                'typprice': talib_typprice,
                'wclprice': talib_wclprice,
                }
        )

        # Need a proper date type
        volume_volatility_cycle_price_df['date'] = pd.to_datetime(
            volume_volatility_cycle_price_df['date'])

        return volume_volatility_cycle_price_df
    
    def talib_get_pattern_recognition_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    # TA-Lib Pattern Recognition indicators
        # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/pattern_recognition.md
        # Nice article about candles (pattern recognition) https://medium.com/analytics-vidhya/recognizing-over-50-candlestick-patterns-with-python-4f02a1822cb5

        # CDL2CROWS - Two Crows
        talib_cdl2crows = talib.CDL2CROWS(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDL3BLACKCROWS - Three Black Crows
        talib_cdl3blackrows = talib.CDL3BLACKCROWS(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDL3INSIDE - Three Inside Up/Down
        talib_cdl3inside = talib.CDL3INSIDE(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDL3LINESTRIKE - Three-Line Strike
        talib_cdl3linestrike = talib.CDL3LINESTRIKE(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDL3OUTSIDE - Three Outside Up/Down
        talib_cdl3outside = talib.CDL3OUTSIDE(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDL3STARSINSOUTH - Three Stars In The South
        talib_cdl3starsinsouth = talib.CDL3STARSINSOUTH(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDL3WHITESOLDIERS - Three Advancing White Soldiers
        talib_cdl3whitesoldiers = talib.CDL3WHITESOLDIERS(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLABANDONEDBABY - Abandoned Baby
        talib_cdlabandonedbaby = talib.CDLABANDONEDBABY(
            df.open.values, df.high.values, df.low.values, df.close.values, penetration=0)
        # CDLADVANCEBLOCK - Advance Block
        talib_cdladvancedblock = talib.CDLADVANCEBLOCK(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLBELTHOLD - Belt-hold
        talib_cdlbelthold = talib.CDLBELTHOLD(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLBREAKAWAY - Breakaway
        talib_cdlbreakaway = talib.CDLBREAKAWAY(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLCLOSINGMARUBOZU - Closing Marubozu
        talib_cdlclosingmarubozu = talib.CDLCLOSINGMARUBOZU(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLCONCEALBABYSWALL - Concealing Baby Swallow
        talib_cdlconcealbabyswall = talib.CDLCONCEALBABYSWALL(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLCOUNTERATTACK - Counterattack
        talib_cdlcounterattack = talib.CDLCOUNTERATTACK(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLDARKCLOUDCOVER - Dark Cloud Cover
        talib_cdldarkcloudcover = talib.CDLDARKCLOUDCOVER(
            df.open.values, df.high.values, df.low.values, df.close.values, penetration=0)
        # CDLDOJI - Doji
        talib_cdldoji = talib.CDLDOJI(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLDOJISTAR - Doji Star
        talib_cdldojistar = talib.CDLDOJISTAR(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLDRAGONFLYDOJI - Dragonfly Doji
        talib_cdldragonflydoji = talib.CDLDRAGONFLYDOJI(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLENGULFING - Engulfing Pattern
        talib_cdlengulfing = talib.CDLENGULFING(
            df.open.values, df.high.values, df.low.values, df.close.values)

        # CDLEVENINGDOJISTAR - Evening Doji Star
        talib_cdleveningdojistar = talib.CDLEVENINGDOJISTAR(
            df.open.values, df.high.values, df.low.values, df.close.values, penetration=0)
        # CDLEVENINGSTAR - Evening Star
        talib_cdleveningstar = talib.CDLEVENINGSTAR(
            df.open.values, df.high.values, df.low.values, df.close.values, penetration=0)
        # CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines
        talib_cdlgapsidesidewhite = talib.CDLGAPSIDESIDEWHITE(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLGRAVESTONEDOJI - Gravestone Doji
        talib_cdlgravestonedoji = talib.CDLGRAVESTONEDOJI(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLHAMMER - Hammer
        talib_cdlhammer = talib.CDLHAMMER(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLHANGINGMAN - Hanging Man
        talib_cdlhangingman = talib.CDLHANGINGMAN(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLHARAMI - Harami Pattern
        talib_cdlharami = talib.CDLHARAMI(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLHARAMICROSS - Harami Cross Pattern
        talib_cdlharamicross = talib.CDLHARAMICROSS(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLHIGHWAVE - high-Wave Candle
        talib_cdlhighwave = talib.CDLHIGHWAVE(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLHIKKAKE - Hikkake Pattern
        talib_cdlhikkake = talib.CDLHIKKAKE(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLHIKKAKEMOD - Modified Hikkake Pattern
        talib_cdlhikkakemod = talib.CDLHIKKAKEMOD(
            df.open.values, df.high.values, df.low.values, df.close.values)

        # CDLHOMINGPIGEON - Homing Pigeon
        talib_cdlhomingpigeon = talib.CDLHOMINGPIGEON(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLIDENTICAL3CROWS - Identical Three Crows
        talib_cdlidentical3crows = talib.CDLIDENTICAL3CROWS(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLINNECK - In-Neck Pattern
        talib_cdlinneck = talib.CDLINNECK(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLINVERTEDHAMMER - Inverted Hammer
        talib_cdlinvertedhammer = talib.CDLINVERTEDHAMMER(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLKICKING - Kicking
        talib_cdlkicking = talib.CDLKICKING(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu
        talib_cdlkickingbylength = talib.CDLKICKINGBYLENGTH(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLLADDERBOTTOM - Ladder Bottom
        talib_cdlladderbottom = talib.CDLLADDERBOTTOM(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLLONGLEGGEDDOJI - Long Legged Doji
        talib_cdllongleggeddoji = talib.CDLLONGLEGGEDDOJI(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLLONGLINE - Long Line Candle
        talib_cdllongline = talib.CDLLONGLINE(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLMARUBOZU - Marubozu
        talib_cdlmarubozu = talib.CDLMARUBOZU(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLMATCHINGLOW - Matching low
        talib_cdlmatchinglow = talib.CDLMATCHINGLOW(
            df.open.values, df.high.values, df.low.values, df.close.values)

        # CDLMATHOLD - Mat Hold
        talib_cdlmathold = talib.CDLMATHOLD(
            df.open.values, df.high.values, df.low.values, df.close.values, penetration=0)
        # CDLMORNINGDOJISTAR - Morning Doji Star
        talib_cdlmorningdojistar = talib.CDLMORNINGDOJISTAR(
            df.open.values, df.high.values, df.low.values, df.close.values, penetration=0)
        # CDLMORNINGSTAR - Morning Star
        talib_cdlmorningstar = talib.CDLMORNINGSTAR(
            df.open.values, df.high.values, df.low.values, df.close.values, penetration=0)
        # CDLONNECK - On-Neck Pattern
        talib_cdlonneck = talib.CDLONNECK(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLPIERCING - Piercing Pattern
        talib_cdlpiercing = talib.CDLPIERCING(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLRICKSHAWMAN - Rickshaw Man
        talib_cdlrickshawman = talib.CDLRICKSHAWMAN(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLRISEFALL3METHODS - Rising/Falling Three Methods
        talib_cdlrisefall3methods = talib.CDLRISEFALL3METHODS(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLSEPARATINGLINES - Separating Lines
        talib_cdlseparatinglines = talib.CDLSEPARATINGLINES(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLSHOOTINGSTAR - Shooting Star
        talib_cdlshootingstar = talib.CDLSHOOTINGSTAR(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLSHORTLINE - Short Line Candle
        talib_cdlshortline = talib.CDLSHORTLINE(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLSPINNINGTOP - Spinning Top
        talib_cdlspinningtop = talib.CDLSPINNINGTOP(
            df.open.values, df.high.values, df.low.values, df.close.values)

        # CDLSTALLEDPATTERN - Stalled Pattern
        talib_cdlstalledpattern = talib.CDLSTALLEDPATTERN(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLSTICKSANDWICH - Stick Sandwich
        talib_cdlsticksandwich = talib.CDLSTICKSANDWICH(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLTAKURI - Takuri (Dragonfly Doji with very long lower shadow)
        talib_cdltakuru = talib.CDLTAKURI(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLTASUKIGAP - Tasuki Gap
        talib_cdltasukigap = talib.CDLTASUKIGAP(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLTHRUSTING - Thrusting Pattern
        talib_cdlthrusting = talib.CDLTHRUSTING(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLTRISTAR - Tristar Pattern
        talib_cdltristar = talib.CDLTRISTAR(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLUNIQUE3RIVER - Unique 3 River
        talib_cdlunique3river = talib.CDLUNIQUE3RIVER(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLUPSIDEGAP2CROWS - Upside Gap Two Crows
        talib_cdlupsidegap2crows = talib.CDLUPSIDEGAP2CROWS(
            df.open.values, df.high.values, df.low.values, df.close.values)
        # CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods
        talib_cdlxsidegap3methods = talib.CDLXSIDEGAP3METHODS(
            df.open.values, df.high.values, df.low.values, df.close.values)

        pattern_indicators_df = pd.DataFrame(
            {'date': df.date.values,
                'symbol': df.symbol,
                # TA-Lib Pattern Recognition indicators
                'cdl2crows': talib_cdl2crows,
                'cdl3blackrows': talib_cdl3blackrows,
                'cdl3inside': talib_cdl3inside,
                'cdl3linestrike': talib_cdl3linestrike,
                'cdl3outside': talib_cdl3outside,
                'cdl3starsinsouth': talib_cdl3starsinsouth,
                'cdl3whitesoldiers': talib_cdl3whitesoldiers,
                'cdlabandonedbaby': talib_cdlabandonedbaby,
                'cdladvancedblock': talib_cdladvancedblock,
                'cdlbelthold': talib_cdlbelthold,
                'cdlbreakaway': talib_cdlbreakaway,
                'cdlclosingmarubozu': talib_cdlclosingmarubozu,
                'cdlconcealbabyswall': talib_cdlconcealbabyswall,
                'cdlcounterattack': talib_cdlcounterattack,
                'cdldarkcloudcover': talib_cdldarkcloudcover,
                'cdldoji': talib_cdldoji,
                'cdldojistar': talib_cdldojistar,
                'cdldragonflydoji': talib_cdldragonflydoji,
                'cdlengulfing': talib_cdlengulfing,
                'cdleveningdojistar': talib_cdleveningdojistar,
                'cdleveningstar': talib_cdleveningstar,
                'cdlgapsidesidewhite': talib_cdlgapsidesidewhite,
                'cdlgravestonedoji': talib_cdlgravestonedoji,
                'cdlhammer': talib_cdlhammer,
                'cdlhangingman': talib_cdlhangingman,
                'cdlharami': talib_cdlharami,
                'cdlharamicross': talib_cdlharamicross,
                'cdlhighwave': talib_cdlhighwave,
                'cdlhikkake': talib_cdlhikkake,
                'cdlhikkakemod': talib_cdlhikkakemod,
                'cdlhomingpigeon': talib_cdlhomingpigeon,
                'cdlidentical3crows': talib_cdlidentical3crows,
                'cdlinneck': talib_cdlinneck,
                'cdlinvertedhammer': talib_cdlinvertedhammer,
                'cdlkicking': talib_cdlkicking,
                'cdlkickingbylength': talib_cdlkickingbylength,
                'cdlladderbottom': talib_cdlladderbottom,
                'cdllongleggeddoji': talib_cdllongleggeddoji,
                'cdllongline': talib_cdllongline,
                'cdlmarubozu': talib_cdlmarubozu,
                'cdlmatchinglow': talib_cdlmatchinglow,
                'cdlmathold': talib_cdlmathold,
                'cdlmorningdojistar': talib_cdlmorningdojistar,
                'cdlmorningstar': talib_cdlmorningstar,
                'cdlonneck': talib_cdlonneck,
                'cdlpiercing': talib_cdlpiercing,
                'cdlrickshawman': talib_cdlrickshawman,
                'cdlrisefall3methods': talib_cdlrisefall3methods,
                'cdlseparatinglines': talib_cdlseparatinglines,
                'cdlshootingstar': talib_cdlshootingstar,
                'cdlshortline': talib_cdlshortline,
                'cdlspinningtop': talib_cdlspinningtop,
                'cdlstalledpattern': talib_cdlstalledpattern,
                'cdlsticksandwich': talib_cdlsticksandwich,
                'cdltakuru': talib_cdltakuru,
                'cdltasukigap': talib_cdltasukigap,
                'cdlthrusting': talib_cdlthrusting,
                'cdltristar': talib_cdltristar,
                'cdlunique3river': talib_cdlunique3river,
                'cdlupsidegap2crows': talib_cdlupsidegap2crows,
                'cdlxsidegap3methods': talib_cdlxsidegap3methods
                }
        )

        # Need a proper date type
        pattern_indicators_df['date'] = pd.to_datetime(
            pattern_indicators_df['date'])

        return pattern_indicators_df
    

class TickerExtender(TechnicalIndicators):
    def calculate_growth_features(self,
                                  df: pd.DataFrame,
                                prefix: str = ""
                                ) -> pd.DataFrame:
        """
        Calculates growth features for a given DataFrame of price data.

        Args:
            df (pd.DataFrame): DataFrame containing at least a 'close' column to calculate growth on.
            prefix (str, optional): A prefix to prepend to the new growth columns (default is "").

        Returns:
            pd.DataFrame: The original DataFrame with additional growth feature columns.

        Notes
        -----
        - The lags are adjusted for a typical trading calendar (e.g., 1d = 1 trading day, 30d = 22 trading days).
        - The growth columns are calculated as `(1 + pct_change(lag))`.
        """
        df = df.copy()

        df.columns = [col.lower() for col in df.columns]

        # adjusted day lags for trading calendar
        trading_day_lags = {
            # 'future_7d': -5, # used for prediction
            '1d': 1,
            '3d': 3,
            '7d': 5,
            '30d': 22,
            '90d': 66,
            '365d': 252
        }


        for lag_name, lag in trading_day_lags.items():
            df[prefix + 'growth_adj_'+ lag_name] = df["close"].pct_change(lag) + 1

        growth_cols = [col for col in df.columns if "growth" in col]

        return df[growth_cols]
    
    def generate_cyclical_features(self, value, period):
        sine = np.sin(2 * np.pi * value / period)
        cosine = np.cos(2 * np.pi * value / period)
        return sine, cosine
    
    
    def add_technical_indicators(self, stocks_df):
        # need to have same 'utc' time on both sides
        # https://stackoverflow.com/questions/73964894/you-are-trying-to-merge-on-datetime64ns-utc-and-datetime64ns-columns-if-yo
        stocks_df['date']= pd.to_datetime(stocks_df['date'], utc=True)

        # Momentum technical indicators
        df_current_ticker_momentum_indicators = self.talib_get_momentum_indicators_for_one_ticker(stocks_df)
        df_current_ticker_momentum_indicators["date"]= pd.to_datetime(df_current_ticker_momentum_indicators['date'], utc=True)

        # Vol
        df_current_ticker_volume_indicators = self.talib_get_volume_volatility_cycle_price_indicators(stocks_df)
        df_current_ticker_volume_indicators["date"]= pd.to_datetime(df_current_ticker_volume_indicators['date'], utc=True)

        df_current_ticker_pattern_indicators = self.talib_get_pattern_recognition_indicators(stocks_df)
        df_current_ticker_pattern_indicators["date"]= pd.to_datetime(df_current_ticker_pattern_indicators['date'], utc=True)

        # merge to one df
        m1 = pd.merge(stocks_df, df_current_ticker_momentum_indicators.reset_index(), how = 'left', on = ["date","symbol"], validate = "one_to_one")
        m2 = pd.merge(m1, df_current_ticker_volume_indicators.reset_index(), how = 'left', on = ["date","symbol"], validate = "one_to_one")
        stocks_with_tech_ind = pd.merge(m2, df_current_ticker_pattern_indicators.reset_index(), how = 'left', on = ["date","symbol"], validate = "one_to_one")

        return stocks_with_tech_ind
    


    def compute_daily_ticker_features(self, historical_prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes a variety of time-based, technical, and growth-related features for daily ticker data.

        Args:
            historical_prices_df (pd.DataFrame): 
                The input DataFrame containing historical price data. Required columns include:
                - 'open': opening price of the asset.
                - 'high': highest price of the asset during the day.
                - 'low': lowest price of the asset during the day.
                - 'close': Closing price of the asset.
                The DataFrame's index should be a dateTimeIndex representing trading days.

        Returns:
            pd.DataFrame: 
                The input DataFrame enriched with additional feature columns, including:
                - date-related features: 'year', 'month', 'weekday', 'date'.
                - Growth metrics for various trading day lags (via `calculate_growth_features`).
                - Simple moving averages: 'SMA10', 'SMA20'.
                - Moving average trend: 'is_growing_moving_average'.
                - Relative daily spread: 'high_minus_low_relative'.
                - Rolling 30-day volatility: '30d_volatility'.
                - Binary 7-day growth signal: 'is_positive_growth_7d'.
        """
        historical_prices_df.columns = [col.lower() for col in historical_prices_df.columns]

        # convert original features to float
        for col in historical_prices_df.select_dtypes(np.number).columns:
            historical_prices_df[col] = historical_prices_df[col].astype("float")


        # date features
        historical_prices_df['year'] = historical_prices_df.index.year
        historical_prices_df['month'] = historical_prices_df.index.month
        historical_prices_df['weekday'] = historical_prices_df.index.weekday
        historical_prices_df['quarter_n'] = historical_prices_df.index.quarter

        # filter warning of timezone info lost
        warnings.filterwarnings("ignore")
        historical_prices_df['quarter'] = historical_prices_df.index.to_period('Q').to_timestamp()
        historical_prices_df['month_dt'] = historical_prices_df.index.to_period('M').to_timestamp()
        warnings.filterwarnings("default") # restore warnings

        historical_prices_df['date'] = historical_prices_df.index.date

        # growth features
        historical_prices_df = pd.concat([historical_prices_df,
                                         self.calculate_growth_features(historical_prices_df)],axis=1)
        # historical_prices_df = self.calculate_growth_features(historical_prices_df)


        # Simple moving averages
        historical_prices_df['SMA10'] = historical_prices_df['close'].rolling(10).mean()
        historical_prices_df['SMA20'] = historical_prices_df['close'].rolling(20).mean()
        historical_prices_df['is_growing_moving_average'] = np.where(
            historical_prices_df['SMA10'] > historical_prices_df['SMA20'], 1, 0
        )

        # daily spread relative to close
        historical_prices_df['high_minus_low_relative'] = (
            (historical_prices_df["high"] - historical_prices_df["low"])
            / historical_prices_df['close']
        )

        # 30d rolling volatility
        historical_prices_df['30d_volatility'] = (
            historical_prices_df['close'].rolling(30).std()
            * np.sqrt(252)
        )

        # continuous growth in 7d
        historical_prices_df['growth_adj_future_7d'] = 1 - historical_prices_df['close'].pct_change(-5) 

        # binary growth in 7d
        historical_prices_df['is_positive_growth_7d'] = np.where(
            historical_prices_df['growth_adj_future_7d'] > 1,
            1,
            0
        )

        # continuous growth in 30d
        historical_prices_df['growth_adj_future_30d'] = 1 - historical_prices_df['close'].pct_change(-22) 

        # binary growth in 30d
        historical_prices_df['is_positive_growth_30d'] = np.where(
            historical_prices_df['growth_adj_future_30d'] > 1,
            1,
            0
        )

        historical_prices_df_tech_indicators = self.add_technical_indicators(historical_prices_df)

        return historical_prices_df_tech_indicators
    

    def transform_euro_yield_df(self, eurostat_euro_yield_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the Eurostat Euro yield DataFrame into a more usable format for analysis.

        Args:
            eurostat_euro_yield_df (pd.DataFrame): 
                DataFrame containing Eurostat Euro yield data. The input DataFrame should have columns 
                for 'maturity', 'variable', and the yield data for various time periods.

        Returns:
            pd.DataFrame: 
                A transformed DataFrame where the yields are pivoted and columns are renamed with a 'eur_yld_' prefix.
                The index represents dates and columns represent maturities for the Euro yield data.

        Notes
        -----
            - The input data is reshaped from a long format to a wide format.
        """
        

        eurostat_euro_yield_df = (pd.melt(frame=eurostat_euro_yield_df,
                                        id_vars=eurostat_euro_yield_df.columns[1:4], 
                                        value_vars=eurostat_euro_yield_df.columns[5:])
                                        .pivot(index="variable", columns="maturity", values="value"))

        eurostat_euro_yield_df.columns.name = None
        eurostat_euro_yield_df.index.name = "date"

        eurostat_euro_yield_df.index = pd.to_datetime(eurostat_euro_yield_df.index, utc=True, format="%Y-%m-%d")

        eurex = mcal.get_calendar('EUREX') 
        new_index = eurex.schedule(start_date=eurostat_euro_yield_df.index[-1]+pd.DateOffset(days=1), 
                                   end_date=eurostat_euro_yield_df.index[-1]+pd.DateOffset(days=2)).index

        eurostat_euro_yield_df = pd.concat([eurostat_euro_yield_df,pd.DataFrame(columns=eurostat_euro_yield_df.columns, index=new_index)],axis=0).shift(2)

        eurostat_euro_yield_df.index = pd.to_datetime(eurostat_euro_yield_df.index, utc=True, format="%Y-%m-%d")

        eurostat_euro_yield_df = eurostat_euro_yield_df.iloc[2:,:]

        eurostat_euro_yield_df.columns = ["eur_yld_" + col + "_prev_2d" for col in eurostat_euro_yield_df.columns]

        return eurostat_euro_yield_df

    def transform_daily_tickers_parallel(self, dir_path: str) -> None:
        """
        Applies the daily ticker transformation to all CSV files in the specified directory in parallel. 
        Then, the transformed files are saved in the corresponding "transformed" directory.

        Args:
            dir_path (str): The directory path where the CSV files containing historical price data are stored.

        Returns:
            List: List with all transformed datasets.

        Notes
        -----
            - This function processes multiple files in parallel for faster execution.
            - Each CSV file in the directory will be processed by the `compute_daily_ticker_features` method.
            - The transformed CSV files will be saved in the corresponding "transformed" directory to 
            the "extracted" of the original files.
        """
        results = Parallel(n_jobs=-1)(
            delayed(self.read_transform_save)(self.compute_daily_ticker_features,str(ticker_file_path))
            for ticker_file_path in self.list_all_files(dir_path) if ticker_file_path.suffix == ".csv"
        )

        return results
    
    def merge_tickers(self, ticker_df_list: List, verbose: Optional[bool] = False):
        merged_tickers_df = pd.DataFrame()

        for ticker_df in tqdm(ticker_df_list):
            # print progress
            if verbose:
                ticker_name = ticker_df["symbol"].unique()[0]
                tqdm.write(f"Current ticker being processed is: {ticker_name}")

            merged_tickers_df = pd.concat([merged_tickers_df,ticker_df], ignore_index = False)
        
        return merged_tickers_df


class MacroIndicatorTransformer():
    pass