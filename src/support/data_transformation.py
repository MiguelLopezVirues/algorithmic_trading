import os
import polars as pl
import polars.selectors as cs

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import talib
from typing import List, Optional, Callable
from functools import partial

from .file_handling import FileHandler

from datetime import datetime

from typing import Tuple, Dict

    
class TechnicalIndicators(FileHandler):
    def talib_get_momentum_indicators_for_one_ticker(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Computes momentum indicators for a single ticker using TA-Lib.

        Args:
            df (pl.DataFrame): DataFrame containing OHLCV data for a single ticker.

        Returns:
            pl.DataFrame: DataFrame with added momentum indicators.
        """
        # Convert Polars Series to numpy arrays for TA-Lib
        high = df['high'].to_numpy().astype('float64')
        low = df['low'].to_numpy().astype('float64')
        close = df['close'].to_numpy().astype('float64')
        open = df['open'].to_numpy().astype('float64')
        volume = df['volume'].to_numpy().astype('float64')
        # ADX - Average Directional Movement Index
        talib_momentum_adx = talib.ADX(high, low, close, timeperiod=14)
        # ADXR - Average Directional Movement Index Rating
        talib_momentum_adxr = talib.ADXR(high, low, close, timeperiod=14 )
        # APO - Absolute Price Oscillator
        talib_momentum_apo = talib.APO(close, fastperiod=12, slowperiod=26, matype=0 )
        # AROON - Aroon
        talib_momentum_aroon = talib.AROON(high, low, timeperiod=14 )
        # talib_momentum_aroon[0].size
        # talib_momentum_aroon[1].size
        # AROONOSC - Aroon Oscillator
        talib_momentum_aroonosc = talib.AROONOSC(high, low, timeperiod=14)
        # BOP - Balance of Power
        # https://school.stockcharts.com/doku.php?id=technical_indicators:balance_of_power
        #calculate open prices as shifted closed prices from the prev day
        # open = df.Last.shift(1)
        talib_momentum_bop = talib.BOP(open, high, low, close)
        # CCI - Commodity Channel Index
        talib_momentum_cci = talib.CCI(high, low, close, timeperiod=14)
        # CMO - Chande Momentum Oscillator
        talib_momentum_cmo = talib.CMO(close, timeperiod=14)
        # DX - Directional Movement Index
        talib_momentum_dx = talib.DX(high, low, close, timeperiod=14)
        # MACD - Moving Average Convergence/Divergence
        talib_momentum_macd, talib_momentum_macdsignal, talib_momentum_macdhist = talib.MACD(close, fastperiod=12, \
                                                                                slowperiod=26, signalperiod=9)
        # MACDEXT - MACD with controllable MA type
        talib_momentum_macd_ext, talib_momentum_macdsignal_ext, talib_momentum_macdhist_ext = talib.MACDEXT(close, \
                                                                                            fastperiod=12, \
                                                                                            fastmatype=0, \
                                                                                            slowperiod=26, \
                                                                                            slowmatype=0, \
                                                                                            signalperiod=9, \
                                                                                            signalmatype=0)
        # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        talib_momentum_macd_fix, talib_momentum_macdsignal_fix, talib_momentum_macdhist_fix = talib.MACDFIX(close, \
                                                                                                signalperiod=9)
        # MFI - Money Flow Index
        talib_momentum_mfi = talib.MFI(high, low, close, volume, timeperiod=14)
        # MINUS_DI - Minus Directional Indicator
        talib_momentum_minus_di = talib.MINUS_DM(high, low, timeperiod=14)
        # MOM - Momentum
        talib_momentum_mom = talib.MOM(close, timeperiod=10)
        # PLUS_DI - Plus Directional Indicator
        talib_momentum_plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
        # PLUS_DM - Plus Directional Movement
        talib_momentum_plus_dm = talib.PLUS_DM(high, low, timeperiod=14)
        # PPO - Percentage Price Oscillator
        talib_momentum_ppo = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
        # ROC - Rate of change : ((price/prevPrice)-1)*100
        talib_momentum_roc = talib.ROC(close, timeperiod=10)
        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        talib_momentum_rocp = talib.ROCP(close, timeperiod=10)
        # ROCR - Rate of change ratio: (price/prevPrice)
        talib_momentum_rocr = talib.ROCR(close, timeperiod=10)
        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        talib_momentum_rocr100 = talib.ROCR100(close, timeperiod=10)
        # RSI - Relative Strength Index
        talib_momentum_rsi = talib.RSI(close, timeperiod=14)
        # STOCH - Stochastic
        talib_momentum_slowk, talib_momentum_slowd = talib.STOCH(high, low, close, \
                                                    fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        # STOCHF - Stochastic Fast
        talib_momentum_fastk, talib_momentum_fastd = talib.STOCHF(high, low, close, \
                                                    fastk_period=5, fastd_period=3, fastd_matype=0)
        # STOCHRSI - Stochastic Relative Strength Index
        talib_momentum_fastk_rsi, talib_momentum_fastd_rsi = talib.STOCHRSI(close, timeperiod=14, \
                                                                fastk_period=5, fastd_period=3, fastd_matype=0)
        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        talib_momentum_trix = talib.TRIX(close, timeperiod=30)
        # ULTOSC - Ultimate Oscillator
        talib_momentum_ultosc = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        # WILLR - Williams' %R
        talib_momentum_willr = talib.WILLR(high, low, close, timeperiod=14)

        # Create Polars DataFrame with all indicators
        momentum_df = pl.DataFrame({
            'datetime': df['datetime'],
            'symbol': df['symbol'],
            'adx': talib_momentum_adx,
            'adxr': talib_momentum_adxr,
            'apo': talib_momentum_apo,
            'aroon_1': talib_momentum_aroon[0],
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
            'willr': talib_momentum_willr
        })
        return momentum_df
    
    def talib_get_volume_volatility_cycle_price_indicators(self,df: pl.DataFrame) -> pl.DataFrame:
        """
        Computes volume, volatility, cycle, and price indicators for a single ticker using TA-Lib.

        Args:
            df (pl.DataFrame): DataFrame containing OHLCV data for a single ticker.

        Returns:
            pl.DataFrame: DataFrame with added volume, volatility, cycle, and price indicators.
        """
        high = df['high'].to_numpy().astype('float64')
        low = df['low'].to_numpy().astype('float64')
        close = df['close'].to_numpy().astype('float64')
        volume = df['volume'].to_numpy().astype('float64')
        open = df['open'].to_numpy().astype('float64')

        # TA-Lib volume indicators
        # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/volume_indicators.md
        # AD - Chaikin A/D Line
        talib_ad = talib.AD(high, low, close, volume)
        # ADOSC - Chaikin A/D Oscillator
        talib_adosc = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        # OBV - On Balance volume
        talib_obv = talib.OBV( close, volume)

        # TA-Lib Volatility indicators
        # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/volatility_indicators.md
        # ATR - Average True Range
        talib_atr = talib.ATR(high, low, close, timeperiod=14)
        # NATR - Normalized Average True Range
        talib_natr = talib.NATR(high, low, close, timeperiod=14)
        # OBV - On Balance volume
        talib_obv = talib.OBV(close, volume)

        # TA-Lib Cycle Indicators
        # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/cycle_indicators.md
        # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
        talib_ht_dcperiod = talib.HT_DCPERIOD(close)
        # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
        talib_ht_dcphase = talib.HT_DCPHASE(close)
        # HT_PHASOR - Hilbert Transform - Phasor Components
        talib_ht_phasor_inphase, talib_ht_phasor_quadrature = talib.HT_PHASOR(close)
        # HT_SINE - Hilbert Transform - SineWave
        talib_ht_sine_sine, talib_ht_sine_leadsine = talib.HT_SINE(close)
        # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
        talib_ht_trendmode = talib.HT_TRENDMODE(close)

        # TA-Lib Price Transform Functions
        # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/price_transform.md
        # AVGPRICE - Average Price
        talib_avgprice = talib.AVGPRICE(open, high, low, close)
        # MEDPRICE - Median Price
        talib_medprice = talib.MEDPRICE(high, low)
        # TYPPRICE - Typical Price
        talib_typprice = talib.TYPPRICE(high, low, close)
        # WCLPRICE - Weighted close Price
        talib_wclprice = talib.WCLPRICE(high, low, close)

        volume_volatility_cycle_price_df = pl.DataFrame({
            'datetime': df['datetime'],
            'symbol': df['symbol'],
            'ad': talib_ad,
            'adosc': talib_adosc,
            'obv': talib_obv,
            'atr': talib_atr,
            'natr': talib_natr,
            'ht_dcperiod': talib_ht_dcperiod,
            'ht_dcphase': talib_ht_dcphase,
            'ht_phasor_inphase': talib_ht_phasor_inphase,
            'ht_phasor_quadrature': talib_ht_phasor_quadrature,
            'ht_sine_sine': talib_ht_sine_sine,
            'ht_sine_leadsine': talib_ht_sine_leadsine,
            'ht_trendmod': talib_ht_trendmode,
            'avgprice': talib_avgprice,
            'medprice': talib_medprice,
            'typprice': talib_typprice,
            'wclprice': talib_wclprice,
        })
        #.with_columns(pl.col('datetime').str.to_datetime())

        return volume_volatility_cycle_price_df
    
    def talib_get_pattern_recognition_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Computes pattern recognition indicators for a single ticker using TA-Lib.

        Args:
            df (pl.DataFrame): DataFrame containing OHLCV data for a single ticker.

        Returns:
            pl.DataFrame: DataFrame with added pattern recognition indicators.
        """

        open = df['open'].to_numpy().astype('float64')
        high = df['high'].to_numpy().astype('float64')
        low = df['low'].to_numpy().astype('float64')
        close = df['close'].to_numpy().astype('float64')
        # TA-Lib Pattern Recognition indicators
        # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/pattern_recognition.md
        # Nice article about candles (pattern recognition) https://medium.com/analytics-vidhya/recognizing-over-50-candlestick-patterns-with-python-4f02a1822cb5

        # CDL2CROWS - Two Crows
        talib_cdl2crows = talib.CDL2CROWS(
            open, high, low, close)
        # CDL3BLACKCROWS - Three Black Crows
        talib_cdl3blackrows = talib.CDL3BLACKCROWS(
            open, high, low, close)
        # CDL3INSIDE - Three Inside Up/Down
        talib_cdl3inside = talib.CDL3INSIDE(
            open, high, low, close)
        # CDL3LINESTRIKE - Three-Line Strike
        talib_cdl3linestrike = talib.CDL3LINESTRIKE(
            open, high, low, close)
        # CDL3OUTSIDE - Three Outside Up/Down
        talib_cdl3outside = talib.CDL3OUTSIDE(
            open, high, low, close)
        # CDL3STARSINSOUTH - Three Stars In The South
        talib_cdl3starsinsouth = talib.CDL3STARSINSOUTH(
            open, high, low, close)
        # CDL3WHITESOLDIERS - Three Advancing White Soldiers
        talib_cdl3whitesoldiers = talib.CDL3WHITESOLDIERS(
            open, high, low, close)
        # CDLABANDONEDBABY - Abandoned Baby
        talib_cdlabandonedbaby = talib.CDLABANDONEDBABY(
            open, high, low, close, penetration=0)
        # CDLADVANCEBLOCK - Advance Block
        talib_cdladvancedblock = talib.CDLADVANCEBLOCK(
            open, high, low, close)
        # CDLBELTHOLD - Belt-hold
        talib_cdlbelthold = talib.CDLBELTHOLD(
            open, high, low, close)
        # CDLBREAKAWAY - Breakaway
        talib_cdlbreakaway = talib.CDLBREAKAWAY(
            open, high, low, close)
        # CDLCLOSINGMARUBOZU - Closing Marubozu
        talib_cdlclosingmarubozu = talib.CDLCLOSINGMARUBOZU(
            open, high, low, close)
        # CDLCONCEALBABYSWALL - Concealing Baby Swallow
        talib_cdlconcealbabyswall = talib.CDLCONCEALBABYSWALL(
            open, high, low, close)
        # CDLCOUNTERATTACK - Counterattack
        talib_cdlcounterattack = talib.CDLCOUNTERATTACK(
            open, high, low, close)
        # CDLDARKCLOUDCOVER - Dark Cloud Cover
        talib_cdldarkcloudcover = talib.CDLDARKCLOUDCOVER(
            open, high, low, close, penetration=0)
        # CDLDOJI - Doji
        talib_cdldoji = talib.CDLDOJI(
            open, high, low, close)
        # CDLDOJISTAR - Doji Star
        talib_cdldojistar = talib.CDLDOJISTAR(
            open, high, low, close)
        # CDLDRAGONFLYDOJI - Dragonfly Doji
        talib_cdldragonflydoji = talib.CDLDRAGONFLYDOJI(
            open, high, low, close)
        # CDLENGULFING - Engulfing Pattern
        talib_cdlengulfing = talib.CDLENGULFING(
            open, high, low, close)

        # CDLEVENINGDOJISTAR - Evening Doji Star
        talib_cdleveningdojistar = talib.CDLEVENINGDOJISTAR(
            open, high, low, close, penetration=0)
        # CDLEVENINGSTAR - Evening Star
        talib_cdleveningstar = talib.CDLEVENINGSTAR(
            open, high, low, close, penetration=0)
        # CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines
        talib_cdlgapsidesidewhite = talib.CDLGAPSIDESIDEWHITE(
            open, high, low, close)
        # CDLGRAVESTONEDOJI - Gravestone Doji
        talib_cdlgravestonedoji = talib.CDLGRAVESTONEDOJI(
            open, high, low, close)
        # CDLHAMMER - Hammer
        talib_cdlhammer = talib.CDLHAMMER(
            open, high, low, close)
        # CDLHANGINGMAN - Hanging Man
        talib_cdlhangingman = talib.CDLHANGINGMAN(
            open, high, low, close)
        # CDLHARAMI - Harami Pattern
        talib_cdlharami = talib.CDLHARAMI(
            open, high, low, close)
        # CDLHARAMICROSS - Harami Cross Pattern
        talib_cdlharamicross = talib.CDLHARAMICROSS(
            open, high, low, close)
        # CDLHIGHWAVE - high-Wave Candle
        talib_cdlhighwave = talib.CDLHIGHWAVE(
            open, high, low, close)
        # CDLHIKKAKE - Hikkake Pattern
        talib_cdlhikkake = talib.CDLHIKKAKE(
            open, high, low, close)
        # CDLHIKKAKEMOD - Modified Hikkake Pattern
        talib_cdlhikkakemod = talib.CDLHIKKAKEMOD(
            open, high, low, close)

        # CDLHOMINGPIGEON - Homing Pigeon
        talib_cdlhomingpigeon = talib.CDLHOMINGPIGEON(
            open, high, low, close)
        # CDLIDENTICAL3CROWS - Identical Three Crows
        talib_cdlidentical3crows = talib.CDLIDENTICAL3CROWS(
            open, high, low, close)
        # CDLINNECK - In-Neck Pattern
        talib_cdlinneck = talib.CDLINNECK(
            open, high, low, close)
        # CDLINVERTEDHAMMER - Inverted Hammer
        talib_cdlinvertedhammer = talib.CDLINVERTEDHAMMER(
            open, high, low, close)
        # CDLKICKING - Kicking
        talib_cdlkicking = talib.CDLKICKING(
            open, high, low, close)
        # CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu
        talib_cdlkickingbylength = talib.CDLKICKINGBYLENGTH(
            open, high, low, close)
        # CDLLADDERBOTTOM - Ladder Bottom
        talib_cdlladderbottom = talib.CDLLADDERBOTTOM(
            open, high, low, close)
        # CDLLONGLEGGEDDOJI - Long Legged Doji
        talib_cdllongleggeddoji = talib.CDLLONGLEGGEDDOJI(
            open, high, low, close)
        # CDLLONGLINE - Long Line Candle
        talib_cdllongline = talib.CDLLONGLINE(
            open, high, low, close)
        # CDLMARUBOZU - Marubozu
        talib_cdlmarubozu = talib.CDLMARUBOZU(
            open, high, low, close)
        # CDLMATCHINGLOW - Matching low
        talib_cdlmatchinglow = talib.CDLMATCHINGLOW(
            open, high, low, close)

        # CDLMATHOLD - Mat Hold
        talib_cdlmathold = talib.CDLMATHOLD(
            open, high, low, close, penetration=0)
        # CDLMORNINGDOJISTAR - Morning Doji Star
        talib_cdlmorningdojistar = talib.CDLMORNINGDOJISTAR(
            open, high, low, close, penetration=0)
        # CDLMORNINGSTAR - Morning Star
        talib_cdlmorningstar = talib.CDLMORNINGSTAR(
            open, high, low, close, penetration=0)
        # CDLONNECK - On-Neck Pattern
        talib_cdlonneck = talib.CDLONNECK(
            open, high, low, close)
        # CDLPIERCING - Piercing Pattern
        talib_cdlpiercing = talib.CDLPIERCING(
            open, high, low, close)
        # CDLRICKSHAWMAN - Rickshaw Man
        talib_cdlrickshawman = talib.CDLRICKSHAWMAN(
            open, high, low, close)
        # CDLRISEFALL3METHODS - Rising/Falling Three Methods
        talib_cdlrisefall3methods = talib.CDLRISEFALL3METHODS(
            open, high, low, close)
        # CDLSEPARATINGLINES - Separating Lines
        talib_cdlseparatinglines = talib.CDLSEPARATINGLINES(
            open, high, low, close)
        # CDLSHOOTINGSTAR - Shooting Star
        talib_cdlshootingstar = talib.CDLSHOOTINGSTAR(
            open, high, low, close)
        # CDLSHORTLINE - Short Line Candle
        talib_cdlshortline = talib.CDLSHORTLINE(
            open, high, low, close)
        # CDLSPINNINGTOP - Spinning Top
        talib_cdlspinningtop = talib.CDLSPINNINGTOP(
            open, high, low, close)

        # CDLSTALLEDPATTERN - Stalled Pattern
        talib_cdlstalledpattern = talib.CDLSTALLEDPATTERN(
            open, high, low, close)
        # CDLSTICKSANDWICH - Stick Sandwich
        talib_cdlsticksandwich = talib.CDLSTICKSANDWICH(
            open, high, low, close)
        # CDLTAKURI - Takuri (Dragonfly Doji with very long lower shadow)
        talib_cdltakuru = talib.CDLTAKURI(
            open, high, low, close)
        # CDLTASUKIGAP - Tasuki Gap
        talib_cdltasukigap = talib.CDLTASUKIGAP(
            open, high, low, close)
        # CDLTHRUSTING - Thrusting Pattern
        talib_cdlthrusting = talib.CDLTHRUSTING(
            open, high, low, close)
        # CDLTRISTAR - Tristar Pattern
        talib_cdltristar = talib.CDLTRISTAR(
            open, high, low, close)
        # CDLUNIQUE3RIVER - Unique 3 River
        talib_cdlunique3river = talib.CDLUNIQUE3RIVER(
            open, high, low, close)
        # CDLUPSIDEGAP2CROWS - Upside Gap Two Crows
        talib_cdlupsidegap2crows = talib.CDLUPSIDEGAP2CROWS(
            open, high, low, close)
        # CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods
        talib_cdlxsidegap3methods = talib.CDLXSIDEGAP3METHODS(
            open, high, low, close)

        pattern_indicators_df = pl.DataFrame(
            {
            'datetime': df['datetime'],
            'symbol': df['symbol'],
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
        #.with_columns(pl.col('datetime').str.to_datetime())

        return pattern_indicators_df
    

class TickerExtender(TechnicalIndicators):
    def calculate_growth_features(self,
                                    df: pl.DataFrame,
                                    prefix: str = ""
                                    ) -> pl.DataFrame:
        """
        Calculates growth features for a given DataFrame of price data.

        Args:
            df (pl.DataFrame): DataFrame containing at least a 'close' column to calculate growth on.
            prefix (str, optional): A prefix to prepend to the new growth columns (default is "").

        Returns:
            pl.DataFrame: The original DataFrame with additional growth feature columns.

        Notes
        -----
        - The lags are adjusted for a typical trading calendar (e.g., 1d = 1 trading day, 30d = 22 trading days).
        - The growth columns are calculated as `(1 + pct_change(lag))`.
        """


        # Adjusted day lags for trading calendar
        trading_day_lags = {
            '1d': 1,
            '3d': 3,
            '7d': 5,
            '14d': 10,
            '21d': 15,
            '30d': 22,
            '90d': 66,
            '365d': 252
        }

        # Create one lagged growth feature per lag
        for lag_name, lag in trading_day_lags.items():
            df = df.with_columns(
                (pl.col("close").pct_change(lag) + 1).alias(f"{prefix}growth_adj_{lag_name}")
            )


        growth_cols = df.select([col for col in df.columns if "growth" in col])

        return df.select(growth_cols)
    
    def calculate_lagged_features(self,
                                    df: pl.DataFrame,
                                    prefix: str = "",
                                    growth: bool = False
                                    ) -> pl.DataFrame:
        """
        Calculates growth features for a given DataFrame of price data.

        Args:
            df (pl.DataFrame): DataFrame containing at least a 'close' column to calculate growth on.
            prefix (str, optional): A prefix to prepend to the new growth columns (default is "").

        Returns:
            pl.DataFrame: The original DataFrame with additional growth feature columns.

        Notes
        -----
        - The lags are adjusted for a typical trading calendar (e.g., 1d = 1 trading day, 30d = 22 trading days).
        - The growth columns are calculated as `(1 + pct_change(lag))`.
        """


        # Adjusted day lags for trading calendar
        trading_day_lags = {
            '1d': 1,
            '3d': 3,
            '7d': 5,
            '14d': 10,
            '21d': 15,
            '30d': 22,
            '90d': 66,
            '365d': 252
        }

        # Create one lagged growth feature per lag
        for lag_name, lag in trading_day_lags.items():
            if growth:
                df = df.with_columns(
                    (pl.col("close").pct_change(lag) + 1).alias(f"{prefix}growth_{lag_name}")
                )
            else:
                df = df.with_columns(
                    (pl.col("close").shift(lag) + 1).alias(f"{prefix}lag_{lag_name}")
                )

        return df
    
    def calculate_lagged_features_pipe(self,
                                    df: pl.DataFrame,
                                    prefix: str = "",
                                    lags: List[int] = [],
                                    add_default: bool = False,
                                    default_lags: List[int] = [5,10,22,66],
                                    ticker: bool = False,
                                    growth: bool = False
                                    ) -> pl.DataFrame:
        """
        Calculates growth features for a given DataFrame of price data.

        Args:
            df (pl.DataFrame): DataFrame containing at least a 'close' column to calculate growth on.
            prefix (str, optional): A prefix to prepend to the new growth columns (default is "").

        Returns:
            pl.DataFrame: The original DataFrame with additional growth feature columns.

        Notes
        -----
        - The lags are adjusted for a typical trading calendar (e.g., 1d = 1 trading day, 30d = 22 trading days).
        - The growth columns are calculated as `(1 + pct_change(lag))`.
        """

        if prefix:
            prefix += "_"


        if not lags and add_default:
            lags = list(set(lags + default_lags))

        # Create one lagged growth feature per lag
        for lag in lags:
            if growth:
                df = df.with_columns(
                    (pl.col("close").pct_change(lag)).alias(f"{prefix}{lag}d_lag_growth")
                )
            else: 
                df = df.with_columns(
                    (pl.col("close").shift(lag)).alias(f"{prefix}{lag}d_lag")
                )

        # lag_cols = df.select([col for col in df.columns if "lag" in col])

        return df
    
    def calculate_rolling_features(self, df: pl.DataFrame,
                                    prefix: str = "",
                                    ticker: bool = True
                                    ) -> pl.DataFrame:
        """
        Computes rolling features such as moving averages, volatility, and relative spreads for a given DataFrame.

        Args:
            df (pl.DataFrame): Input DataFrame containing OHLCV data.
            prefix (str, optional): Prefix to add to column names. Defaults to "".
            ticker (bool, optional): Whether to include ticker-specific features. Defaults to True.

        Returns:
            pl.DataFrame: DataFrame with added rolling features.
        """
        df = df.select(
                # Moving averages
                pl.col('close').rolling_mean(10).alias(f"{prefix}SMA10"),
                pl.col('close').rolling_mean(22).alias(f'{prefix}SMA22'),
                pl.col('close').rolling_mean(66).alias(f'{prefix}SMA66'),
                # Rolling 30d volatility annualized
                (pl.col('close').rolling_std(22) * np.sqrt(252)).alias(f'{prefix}30d_volatility'),
                # Spread relative to close price
                ((pl.col("high") - pl.col("low")) / pl.col("close")).alias(f'{prefix}high_minus_low_relative') if ticker else None,
                ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias(f'{prefix}_close_relative_high_low'),
            ).with_columns( # Growing moving average
            (pl.col(f"{prefix}SMA10") > pl.col(f'{prefix}SMA22')).cast(pl.Int8).alias(f'{prefix}is_growing_moving_average')
        ).select(pl.exclude("literal")) # drop the None column

        return df
    
    def calculate_rolling_features_experiment(self, df: pl.DataFrame,
                                                prefix: str = "",
                                                ticker: bool = True
                                                ) -> pl.DataFrame:
        """
        Computes additional rolling features for experimental purposes, including advanced metrics like rolling Sharpe ratio,
        autocorrelation, and quantiles.

        Args:
            df (pl.DataFrame): Input DataFrame containing OHLCV data.
            prefix (str, optional): Prefix to add to column names. Defaults to "".
            ticker (bool, optional): Whether to include ticker-specific features. Defaults to True.

        Returns:
            pl.DataFrame: DataFrame with added experimental rolling features.
        """

        if prefix != "":
            prefix += "_"


        if ticker:
            df = df.select(
                    pl.col('close').rolling_mean(10).alias(f"{prefix}SMA10"),
                    pl.col('close').rolling_mean(22).alias(f'{prefix}SMA22'),
                    pl.col('close').rolling_mean(66).alias(f'{prefix}SMA33'),
                    # Rolling 30d volatility annualized
                    (pl.col('close').rolling_std(22) / pl.col('close').rolling_mean(22)).alias(f'{prefix}30d_volatility'),
                    ((pl.col("high") - pl.col("low")) / pl.col("close")).alias(f'{prefix}high_minus_low_relative'),
                    ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias(f'{prefix}_close_relative_high_low'),
                    pl.col("volume").rolling_sum(5).alias("volume_last_5"),
                    pl.col("volume").rolling_sum(10).alias("volume_last_10"),
                    pl.col("volume").rolling_sum(15).alias("volume_last_15"),
                    pl.col("volume").rolling_sum(22).alias("volume_last_22"),
                    pl.col("volume").rolling_sum(66).alias("volume_last_66"),
                    (pl.col("close").cum_max() / pl.col("close")).alias("pct_ATH"),
                    (pl.col("close").cum_min() / pl.col("close")).alias("pct_ATL"),
                    (pl.col("close").rolling_max(22) / pl.col("close")).alias("pct_1_month_high"),
                    (pl.col("close").rolling_max(66) / pl.col("close")).alias("pct_3_month_high"),
                    (pl.col("close").rolling_max(126) / pl.col("close")).alias("pct_6_month_high"),
                    (pl.col("close").rolling_max(252) / pl.col("close")).alias("pct_1_year_high"),
                    (pl.col("close").rolling_max(1260) / pl.col("close")).alias("pct_5_year_high"),
                    (pl.col("close").rolling_min(22) / pl.col("close")).alias("pct_1_month_low"),
                    (pl.col("close").rolling_min(66) / pl.col("close")).alias("pct_3_month_low"),
                    (pl.col("close").rolling_min(126) / pl.col("close")).alias("pct_6_month_low"),
                    (pl.col("close").rolling_min(252) / pl.col("close")).alias("pct_1_year_low"),
                    (pl.col("close").rolling_min(1260) / pl.col("close")).alias("pct_5_year_low"),
                    pl.col("close").pct_change().rolling_sum(10).alias("10_day_rolling_return"),
                    pl.col("close").pct_change().rolling_sum(22).alias("22_day_rolling_return"),
                    pl.col("close").pct_change().rolling_sum(66).alias("66_day_rolling_return"),
                    pl.col("close").pct_change().rolling_sum(126).alias("126_day_rolling_return"),
                    (pl.col("close").pct_change().rolling_sum(10) / pl.col('close').rolling_std(10)).alias("10_day_rolling_sharpe"),
                    (pl.col("close").pct_change().rolling_sum(22) / pl.col('close').rolling_std(10)).alias("22_day_rolling_sharpe"),
                    (pl.col("close").pct_change().rolling_sum(66) / pl.col('close').rolling_std(10)).alias("66_day_rolling_sharpe"),
                    (pl.col("close").pct_change().rolling_sum(126) / pl.col('close').rolling_std(10)).alias("126_day_rolling_sharpe"),
                    pl.rolling_corr(pl.col("close").pct_change(),pl.col("close").pct_change().shift(1),window_size=10).alias("autocorrelation_lag_1"),
                    pl.rolling_corr(pl.col("close").pct_change(),pl.col("close").pct_change().shift(3),window_size=10).alias("autocorrelation_lag_3"),
                    pl.rolling_corr(pl.col("close").pct_change(),pl.col("close").pct_change().shift(6),window_size=15).alias("autocorrelation_lag_6"),
                    pl.rolling_corr(pl.col("close").pct_change(),pl.col("close").pct_change().shift(9),window_size=15).alias("autocorrelation_lag_9"),
                    pl.rolling_corr(pl.col("close").pct_change(),pl.col("close").pct_change().shift(10),window_size=22).alias("autocorrelation_lag_10"),
                    pl.rolling_corr(pl.col("close").pct_change(),pl.col("close").pct_change().shift(13),window_size=22).alias("autocorrelation_lag_13"),
                    pl.col("close").rolling_quantile(0.05, window_size=5).alias("rolling_5_quantile_0.05"),
                    pl.col("close").rolling_quantile(0.05, window_size=10).alias("rolling_10_quantile_0.05"),
                    pl.col("close").rolling_quantile(0.05, window_size=15).alias("rolling_15_quantile_0.05"),
                    pl.col("close").rolling_quantile(0.95, window_size=5).alias("rolling_5_quantile_0.95"),
                    pl.col("close").rolling_quantile(0.95, window_size=10).alias("rolling_10_quantile_0.95"),
                    pl.col("close").rolling_quantile(0.95, window_size=15).alias("rolling_15_quantile_0.95"),
                    pl.col("close").rolling_skew(5).alias("rolling_skew_5"),
                    pl.col("close").rolling_skew(10).alias("rolling_skew_10"),
                    pl.col("close").rolling_skew(15).alias("rolling_skew_15")
                    ).with_columns( # Growing moving average
                    (pl.col(f"{prefix}SMA10") / pl.col(f'{prefix}SMA22')).alias(f'{prefix}SMA'))
        else: 
            df = df.select(
                        # Moving averages
                        pl.col('close').rolling_mean(10).alias(f"{prefix}SMA10"),
                        pl.col('close').rolling_mean(22).alias(f'{prefix}SMA22'),
                        pl.col('close').rolling_mean(66).alias(f'{prefix}SMA33'),
                        # Rolling 30d volatility annualized
                        (pl.col('close').rolling_std(22) / pl.col('close').rolling_mean(22)).alias(f'{prefix}30d_volatility'),
                        # Spread relative to close price
                    ).with_columns( # Growing moving average
                    (pl.col(f"{prefix}SMA10") / pl.col(f'{prefix}SMA22')).alias(f'{prefix}SMA'))



        return df
    
    def calculate_rolling_features_pipe(self, df: pl.DataFrame,
                                    rolling_lags: List[int] = [],
                                    prefix: str = "",
                                    add_default: bool = False,
                                    default_lags: List[int] = [5,10,22,66],
                                    ticker: bool = False,
                                    ) -> pl.DataFrame:
        """
        Computes rolling features for a given DataFrame using a pipeline approach.

        Args:
            df (pl.DataFrame): Input DataFrame containing OHLCV data.
            rolling_lags (List[int], optional): List of rolling window sizes. Defaults to [].
            prefix (str, optional): Prefix to add to column names. Defaults to "".
            add_default (bool, optional): Whether to add default lags. Defaults to False.
            default_lags (List[int], optional): Default lags to use if `add_default` is True. Defaults to [5, 10, 22, 66].
            ticker (bool, optional): Whether to include ticker-specific features. Defaults to False.

        Returns:
            pl.DataFrame: DataFrame with added rolling features.
        """
        if prefix:
            prefix += "_"

        if not rolling_lags or add_default:
            rolling_lags = list(set(rolling_lags + default_lags))

        close_col = pl.col("close")
        rolling_cols = [close_col
                .rolling_mean(window_size=rolling_lag)  # Compute rolling mean
                .alias(f"{prefix}{rolling_lag}d_window") for rolling_lag in rolling_lags]
        
        df = df.with_columns(rolling_cols)
        
        # rolling_cols = df.select(pl.col([col for col in df.columns if "window" in col]))
        # df.select(pl.exclude("literal")) # drop the None column

        return df
    
    def calculate_lag_rolling_features_pipe(self,
                                            df: pl.DataFrame,
                                            forecast_horizon: int,
                                            lags: List[int] = [],
                                            rolling_lags: List[int] = [],
                                            prefix: str="",
                                            add_default: bool = False,
                                            growth: bool = False,
                                            ticker: bool = False) -> pl.DataFrame:
        """
        Computes lagged and rolling features for a given DataFrame using a pipeline approach.

        Args:
            df (pl.DataFrame): Input DataFrame containing OHLCV data.
            forecast_horizon (int): Number of steps to shift the features.
            lags (List[int], optional): List of lag sizes. Defaults to [].
            rolling_lags (List[int], optional): List of rolling window sizes. Defaults to [].
            prefix (str, optional): Prefix to add to column names. Defaults to "".
            add_default (bool, optional): Whether to add default lags. Defaults to False.
            growth (bool, optional): Whether to compute growth features. Defaults to False.
            ticker (bool, optional): Whether to include ticker-specific features. Defaults to False.

        Returns:
            pl.DataFrame: DataFrame with added lagged and rolling features.
        """
        if growth:
            df = self.calculate_lagged_features_pipe(df, lags=lags,add_default=add_default, prefix=prefix, growth=True)


        df = self.calculate_rolling_features_pipe(df, rolling_lags=rolling_lags,add_default=add_default, prefix=prefix)
        df = df.select("datetime",
                       "close",
                       pl.col([col for col in df.columns if "window" in col or "growth" in col]).shift(forecast_horizon).name.suffix(f"_lagged_{forecast_horizon}"))

        lags = [lag + (forecast_horizon - 1) if lag < forecast_horizon else lag for lag in lags]
        df = self.calculate_lagged_features_pipe(df, lags=lags,add_default=add_default, prefix=prefix)

        if not ticker:
            df = df.rename({"close":prefix})

        return df
    
    
    
    def calculate_date_features(self, df: pl.DataFrame,
                                    prefix: str = ""
                                    ) -> pl.DataFrame:
        """
        Computes date-related features such as year, month, weekday, and quarter.

        Args:
            df (pl.DataFrame): Input DataFrame containing a datetime column.
            prefix (str, optional): Prefix to add to column names. Defaults to "".

        Returns:
            pl.DataFrame: DataFrame with added date-related features.
        """
        df = df.select(
            pl.col('datetime').dt.year().alias('year'),
            pl.col('datetime').dt.month().alias('month'),
            pl.col('datetime').dt.weekday().alias('weekday'),
            pl.col('datetime').dt.quarter().alias('quarter_n'),
            pl.col('datetime').dt.truncate("1mo").alias("month_dt"),

            # build quarter in date form
            pl.date(pl.col('datetime').dt.year(),
                    (pl.col('datetime').dt.quarter() * 3 - 2),  # Maps quarter to starting month (1,4,7,10)
                    1 # First day of month
            ).alias("quarter")
        )

        return df

    
    def generate_cyclical_features(self, value: float, period_length: float) -> Tuple[float, float]:
        """
        Generates cyclical features (sine and cosine) for a given value and period.

        Args:
            value (float): Input value to transform.
            period (float): Period for the cyclical transformation.

        Returns:
            Tuple[float, float]: Sine and cosine values representing the cyclical feature.
        """
        sine = np.sin(2 * np.pi * value / period_length)
        cosine = np.cos(2 * np.pi * value / period_length)
        return sine, cosine
    
    
    def add_technical_indicators(self, stocks_df: pl.DataFrame) -> pl.DataFrame:
        """
        Adds technical indicators (momentum, volume, volatility, cycle, and pattern recognition) to a DataFrame.

        Args:
            stocks_df (pl.DataFrame): Input DataFrame containing OHLCV data.

        Returns:
            pl.DataFrame: DataFrame with added technical indicators.
        """
        # Calculate indicators
        momentum = self.talib_get_momentum_indicators_for_one_ticker(stocks_df)
        volume = self.talib_get_volume_volatility_cycle_price_indicators(stocks_df)
        patterns = self.talib_get_pattern_recognition_indicators(stocks_df)


        stocks_with_tech_ind = (stocks_df.join(momentum, on=['datetime', 'symbol'], how='left')
                                        .join(volume, on=['datetime', 'symbol'], how='left')
                                        .join(patterns, on=['datetime', 'symbol'], how='left'))

        return stocks_with_tech_ind
    
    def compute_daily_index_features(self, 
                                     historical_index_df: pl.DataFrame, 
                                     prefix: str = "") -> pl.DataFrame:
        """
        Computes daily index features including rolling and growth features.

        Args:
            historical_index_df (pl.DataFrame): Input DataFrame containing index data.
            prefix (str, optional): Prefix to add to column names. Defaults to "".

        Returns:
            pl.DataFrame: DataFrame with added index features.
        """

        # Rolling features
        rolling_features = self.calculate_rolling_features(historical_index_df, prefix)

        # Growth features
        growth_features = self.calculate_growth_features(historical_index_df, prefix)

        historical_index_df = pl.concat([historical_index_df[["datetime"]],growth_features,rolling_features], how="horizontal")

        return historical_index_df

    

    def compute_daily_ticker_features(self, historical_prices_df: pl.DataFrame) -> pl.DataFrame:
        """
        Computes a variety of time-based, technical, and growth-related features for daily ticker data.

        Args:
            historical_prices_df (pl.DataFrame): 
                The input DataFrame containing historical price data. Required columns include:
                - 'open': opening price of the asset.
                - 'high': highest price of the asset during the day.
                - 'low': lowest price of the asset during the day.
                - 'close': Closing price of the asset.
                The DataFrame's index should be a dateTimeIndex representing trading days.

        Returns:
            pl.DataFrame: 
                The input DataFrame enriched with additional feature columns, including:
                - date-related features: 'year', 'month', 'weekday', 'datetime'.
                - Growth metrics for various trading day lags (via `calculate_growth_features`).
                - Simple moving averages: 'SMA10', 'SMA20'.
                - Moving average trend: 'is_growing_moving_average'.
                - Relative daily spread: 'high_minus_low_relative'.
                - Rolling 30-day volatility: '30d_volatility'.
                - Binary 7-day growth signal: 'is_positive_growth_7d'.
        """

        # Date features
        date_features = self.calculate_date_features(historical_prices_df)
        historical_prices_df = historical_prices_df.with_columns(date_features)

        # Growth features
        growth_features = self.calculate_growth_features(historical_prices_df)
        historical_prices_df = historical_prices_df.with_columns(growth_features)

        # Moving averages
        rolling_features = self.calculate_rolling_features(historical_prices_df)
        historical_prices_df = historical_prices_df.with_columns(rolling_features)

        historical_prices_df_tech_indicators = self.add_technical_indicators(historical_prices_df)

        return historical_prices_df_tech_indicators
    

    
    def transform_euro_yield_df(self, eurostat_euro_yield_df: pl.DataFrame) -> pl.DataFrame:
        """
        Transforms the Eurostat Euro yield DataFrame into a more usable format for analysis.

        Args:
            eurostat_euro_yield_df (pl.DataFrame): 
                DataFrame containing Eurostat Euro yield data. The input DataFrame should have columns 
                for 'maturity', 'variable', and the yield data for various time periods.

        Returns:
            pl.DataFrame: 
                A transformed DataFrame where the yields are pivoted and columns are renamed with a 'eur_yld_' prefix.
                The index represents dates and columns represent maturities for the Euro yield data.

        Notes
        -----
            - The input data is reshaped from a long format to a wide format.
        """
        # Unpivot and pivot operations
        unpivotted = eurostat_euro_yield_df.unpivot(
            index= eurostat_euro_yield_df.columns[1:4],  # Original id_vars (columns 1-3)
            on = eurostat_euro_yield_df.columns[5:]  # Original value_vars (columns 5+)
        )

        # Pivot to wide format
        pivoted = unpivotted.pivot(
            values="value",
            index="variable",
            on="maturity",
            aggregate_function="first"
        )

        # Clean up and date handling
        euro_yield = (
            pivoted
            .rename({"variable": 'datetime'})
            #.with_columns(pl.col('datetime').str.to_datetime())
        )

        euro_yield = euro_yield.select(pl.col("datetime").str.to_datetime().dt.add_business_days(2), 
                                        pl.exclude("datetime").cast(pl.Float32))
        
        # Rename columns
        euro_yield = euro_yield.rename({
                        col: f"eur_yld_{col}_prev_2d" 
                        for col in euro_yield.columns 
                        if col not in ['datetime']
                    })
        return euro_yield

    def transform_daily_tickers_parallel(self, dir_path: str) -> None:
        """
        Applies the daily ticker transformation to all parquet files in the specified directory in parallel. 
        Then, the transformed files are saved in the corresponding "transformed" directory.

        Args:
            dir_path (str): The directory path where the parquet files containing historical price data are stored.

        Returns:
            List: List with all transformed datasets.

        Notes
        -----
            - This function processes multiple files in parallel for faster execution.
            - Each parquet file in the directory will be processed by the `compute_daily_ticker_features` method.
            - The transformed parquet files will be saved in the corresponding "transformed" directory to 
            the "extracted" of the original files.
        """


        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.read_transform_save)(self.compute_daily_ticker_features,str(ticker_file_path))
            for ticker_file_path in self.list_all_files(dir_path) if ticker_file_path.suffix == ".parquet"
        )

        # results = []
        # for ticker_file_path in self.list_all_files(Path(dir_path)):

        #     if ticker_file_path.suffix == ".parquet":
        #         results.append(self.read_transform_save(self.compute_daily_ticker_features, str(ticker_file_path)))

        return results
    
    def compute_daily_ticker_features_experiment(self, historical_prices_df: pl.DataFrame, forecast_horizon: int) -> pl.DataFrame:
        """
        Computes a variety of time-based, technical, and growth-related features for daily ticker data.

        Args:
            historical_prices_df (pl.DataFrame): 
                The input DataFrame containing historical price data. Required columns include:
                - 'open': opening price of the asset.
                - 'high': highest price of the asset during the day.
                - 'low': lowest price of the asset during the day.
                - 'close': Closing price of the asset.
                The DataFrame's index should be a dateTimeIndex representing trading days.

        Returns:
            pl.DataFrame: 
                The input DataFrame enriched with additional feature columns, including:
                - date-related features: 'year', 'month', 'weekday', 'datetime'.
                - Growth metrics for various trading day lags (via `calculate_growth_features`).
                - Simple moving averages: 'SMA10', 'SMA20'.
                - Moving average trend: 'is_growing_moving_average'.
                - Relative daily spread: 'high_minus_low_relative'.
                - Rolling 30-day volatility: '30d_volatility'.
        """

        # Date features
        # date_features = self.calculate_date_features(historical_prices_df)
        # historical_prices_df = historical_prices_df.with_columns(date_features)

        # Growth features
        growth_features = self.calculate_growth_features(historical_prices_df)

        growth_features = growth_features.select(pl.all().shift(forecast_horizon).name.suffix(f"_lagged_{forecast_horizon}"))

        historical_prices_df = historical_prices_df.with_columns(growth_features)

        # Moving averages
        rolling_features = self.calculate_rolling_features_experiment(historical_prices_df)
        rolling_features = rolling_features.select(pl.all().shift(forecast_horizon).name.suffix(f"_lagged_{forecast_horizon}"))
        historical_prices_df = historical_prices_df.with_columns(rolling_features)

        return historical_prices_df
    
    def transform_daily_tickers_parallel_experiment(self, dir_path: str, forecast_horizon: int) -> None:
        """
        Modification from transform_daily_tickers_parallel: Applies a different set of transformations for notebook experiments.
        Applies the daily ticker transformation to all parquet files in the specified directory in parallel. 
        Then, the transformed files are saved in the corresponding "transformed" directory.

        Args:
            dir_path (str): The directory path where the parquet files containing historical price data are stored.

        Returns:
            List: List with all transformed datasets.

        Notes
        -----
            - This function processes multiple files in parallel for faster execution.
            - Each parquet file in the directory will be processed by the `compute_daily_ticker_features` method.
            - The transformed parquet files will be saved in the corresponding "transformed" directory to 
            the "extracted" of the original files.
        """

        compute_daily_ticker_features_experiment_partial = partial(self.compute_daily_ticker_features_experiment, forecast_horizon=forecast_horizon)

        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.read_transform_save)(compute_daily_ticker_features_experiment_partial,str(ticker_file_path))
            for ticker_file_path in self.list_all_files(dir_path) if ticker_file_path.suffix == ".parquet"
        )



        return results
    
    
    def apply_transformation_parallel(self, dir_path: str, transformation_function: Callable) -> List:
        """
        Applies a transformation function in parallel to all `.parquet` files in the given directory.

        Args:
            dir_path (str): Path to the directory containing `.parquet` files.
            transformation_function (Callable): Function to apply to each file.

        Returns:
            list: List of results from applying the transformation function.
        """
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(transformation_function)(str(ticker_file_path))
            for ticker_file_path in self.list_all_files(dir_path) if ticker_file_path.suffix == ".parquet"
        )
        return results


    def add_empty_row_df(self, df: pl.DataFrame, date_column: str, n_steps: int = 5) -> pl.DataFrame:
        """
        Adds an empty row to a Polars DataFrame with `n_steps` forward business days.

        Args:
            df (pl.DataFrame): Input DataFrame.
            date_column (str): Name of the datetime column.
            n_steps (int, optional): Number of business days to project forward. Defaults to 5.

        Returns:
            pl.DataFrame: DataFrame with the added empty row.
        """
        # Create empty row
        empty_row = pl.DataFrame({col: [None]*n_steps for col in df.columns})

        # Cast each column in the empty row to the appropriate type using the original schema
        empty_row = empty_row.select([pl.col(col).cast(dtype) for col, dtype in df.schema.items()])

        empty_row = empty_row.with_columns(pl.lit(df[-n_steps:][date_column]).dt.add_business_days(n_steps).alias("datetime").cast(df.schema["datetime"]))

        df_with_empty = pl.concat([df, empty_row])

        return df_with_empty
    


    
    def read_add_exog(
        self, file_path: str, n_last: int = 1265, date_column: str = "datetime", n_steps: int = 5
    ) -> pl.DataFrame:
        """
        Generates the rows of exogenous features necessary for prediction.
        Reads a parquet file, resamples it to 'buiness day' frequency, adds as many empty rows as n_steps, 
        and computes new exogenous features for them. 

        Args:
            file_path (str): Path to the parquet file.
            n_last (int, optional): Number of last rows to consider. Defaults to 1265 due to current exog feature configuration.
            date_column (str, optional): Name of the datetime column. Defaults to "datetime".
            n_steps (int, optional): Forecast horizon. Defaults to 5.

        Returns:
            pl.DataFrame: DataFrame with computed exogenous features.
        """

        ticker_df = self.read_parquet_file(file_path)

        ticker_df =  (ticker_df.upsample(time_column=date_column, every="1d") # .asfreq("B") equivalent in Polars
                        .filter(pl.col(date_column).dt.weekday() <= 5)
                        .fill_null(strategy="forward"))

        ticker_df_last = ticker_df[-n_last*2:]

        ticker_df_with_empty_rows = self.add_empty_row_df(ticker_df_last, date_column = date_column, n_steps = n_steps)

        ticker_df_new_exog = self.compute_daily_ticker_features_experiment(ticker_df_with_empty_rows, forecast_horizon=n_steps
                                                                           ).with_columns(cs.temporal() | cs.string()
                                                                            ).fill_null(strategy="forward")[-n_steps:]

        # save symbol name
        symbol = ticker_df_new_exog["symbol"][0]

        # exclude non-exog columns
        ticker_df_new_exog = ticker_df_new_exog.select(pl.exclude(["close","high","low","volume", "open", "symbol","currency","industry","sector","region"]))

        # convert to pandas for skforecast prediction
        ticker_df_new_exog = ticker_df_new_exog.to_pandas().set_index("datetime").asfreq("B")


        return symbol, ticker_df_new_exog

    
    def merge_tickers(self, 
                        ticker_df_list: List[pl.DataFrame], 
                        verbose: Optional[bool] = False, 
                        method: str = "vertical"
                        ) -> pl.DataFrame:
        """
        Merges a list of Polars DataFrames containing ticker data.

        Args:
            ticker_df_list (List[pl.DataFrame]): List of DataFrames to merge.
            verbose (Optional[bool], optional): If True, prints the number of merged tickers. Defaults to False.
            method (str, optional): Merge method, "vertical" for row-wise or "horizontal" for column-wise. Defaults to "vertical".

        Returns:
            pl.DataFrame: The merged DataFrame.
        """

        merged_tickers_df = pl.concat(ticker_df_list, how = method)

        if verbose:
            tickers = merged_tickers_df['symbol'].unique().to_list()
            print(f"Merged {len(tickers)} tickers")

        return merged_tickers_df
    
    
    def generate_exog_dict(self, dir_path: str, n_last: int = 1265, n_steps: int = 5) -> Dict[str, pl.DataFrame]:
        """
        Generates a dictionary of exogenous data for multiple tickers.

        Args:
            dir_path (str): Path to the directory containing ticker files.
            n_last (int, optional): Number of past days to include. Defaults to 1265.
            n_steps (int, optional): Forecast horizon steps. Defaults to 5.

        Returns:
            Dict[str, pl.DataFrame]: A dictionary where keys are ticker symbols and values are DataFrames with exogenous data.
        """

        add_last_exogenous = partial(self.read_add_exog, n_last = n_last, date_column = "datetime", n_steps = n_steps)

        symbols_df_tuple_list = self.apply_transformation_parallel(dir_path = dir_path, transformation_function = add_last_exogenous)

        return {symbol: ticker_df for symbol, ticker_df in symbols_df_tuple_list}




class MacroIndicatorTransformer():
    pass