import os
import polars as pl
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
import re
import talib
from typing import List, Optional
import pandas_market_calendars as mcal
import warnings
from tqdm import tqdm

from .file_handling import FileHandler


# TO-DO:
# [] add error-handling

    
class TechnicalIndicators(FileHandler):
    def talib_get_momentum_indicators_for_one_ticker(self, df: pl.DataFrame) -> pl.DataFrame:
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
            '30d': 22,
            '90d': 66,
            '365d': 252
        }

        # Create one lagged growth feature per lag
        for lag_name, lag in trading_day_lags.items():
            df = df.with_columns(
                (pl.col("close").pct_change(lag) + 1).alias(f"{prefix}growth_adj_{lag_name}")
            )

        growth_cols = pl.col("^growth.*$")

        return df.select(growth_cols)
    
    def generate_cyclical_features(self, value, period):
        sine = np.sin(2 * np.pi * value / period)
        cosine = np.cos(2 * np.pi * value / period)
        return sine, cosine
    
    
    def add_technical_indicators(self, stocks_df: pl.DataFrame) -> pl.DataFrame:

        # Calculate indicators
        momentum = self.talib_get_momentum_indicators_for_one_ticker(stocks_df)
        volume = self.talib_get_volume_volatility_cycle_price_indicators(stocks_df)
        patterns = self.talib_get_pattern_recognition_indicators(stocks_df)


        stocks_with_tech_ind = (stocks_df.join(momentum, on=['datetime', 'symbol'], how='left')
                                        .join(volume, on=['datetime', 'symbol'], how='left')
                                        .join(patterns, on=['datetime', 'symbol'], how='left'))

        return stocks_with_tech_ind
    


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
        historical_prices_df = historical_prices_df.with_columns(
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


        # Growth features
        growth_features = self.calculate_growth_features(historical_prices_df)
        historical_prices_df = historical_prices_df.with_columns(growth_features)

        # Moving averages
        historical_prices_df = historical_prices_df.with_columns(
                # Moving averages
                pl.col('close').rolling_mean(10).alias('SMA10'),
                pl.col('close').rolling_mean(20).alias('SMA20'),
                # Rolling 30d volatility
                (pl.col('close').rolling_std(30) * np.sqrt(252)).alias('30d_volatility'),
                # Spread relative to close price
                ((pl.col("high") - pl.col("low")) / pl.col("close")).alias('high_minus_low_relative')
            ).with_columns( # Growing moving average
            (pl.col('SMA10') > pl.col('SMA20')).cast(pl.Int8).alias('is_growing_moving_average')
        )

        # Future growth calculations
        for days, offset in [('7d', 5), ('30d', 22)]:
            historical_prices_df = historical_prices_df.with_columns(
                (1 - pl.col('close').pct_change(-offset)).alias(f'growth_adj_future_{days}')
                ).with_columns((pl.col(f'growth_adj_future_{days}') > 1).cast(pl.Int8).alias(f'is_positive_growth_{days}')
            )

        historical_prices_df_tech_indicators = self.add_technical_indicators(historical_prices_df)

        return historical_prices_df_tech_indicators
    

    # def transform_euro_yield_df(self, eurostat_euro_yield_df: pl.DataFrame) -> pl.DataFrame:
    #     """
    #     Transforms the Eurostat Euro yield DataFrame into a more usable format for analysis.

    #     Args:
    #         eurostat_euro_yield_df (pl.DataFrame): 
    #             DataFrame containing Eurostat Euro yield data. The input DataFrame should have columns 
    #             for 'maturity', 'variable', and the yield data for various time periods.

    #     Returns:
    #         pl.DataFrame: 
    #             A transformed DataFrame where the yields are pivoted and columns are renamed with a 'eur_yld_' prefix.
    #             The index represents dates and columns represent maturities for the Euro yield data.

    #     Notes
    #     -----
    #         - The input data is reshaped from a long format to a wide format.
    #     """
        

    #     eurostat_euro_yield_df = (pd.melt(frame=eurostat_euro_yield_df,
    #                                     id_vars=eurostat_euro_yield_df.columns[1:4], 
    #                                     value_vars=eurostat_euro_yield_df.columns[5:])
    #                                     .pivot(index="variable", columns="maturity", values="value"))

    #     eurostat_euro_yield_df.columns.name = None
    #     eurostat_euro_yield_df.index.name = 'datetime'

    #     eurostat_euro_yield_df.index = pd.to_datetime(eurostat_euro_yield_df.index, utc=True, )

    #     eurex = mcal.get_calendar('EUREX') 
    #     new_index = eurex.schedule(start_date=eurostat_euro_yield_df.index[-1]+pd.DateOffset(days=1), 
    #                                end_date=eurostat_euro_yield_df.index[-1]+pd.DateOffset(days=2)).index

    #     eurostat_euro_yield_df = pd.concat([eurostat_euro_yield_df,pl.DataFrame(columns=eurostat_euro_yield_df.columns, index=new_index)],axis=0).shift(2)

    #     eurostat_euro_yield_df.index = pd.to_datetime(eurostat_euro_yield_df.index, utc=True, )

    #     eurostat_euro_yield_df = eurostat_euro_yield_df.iloc[2:,:]

    #     eurostat_euro_yield_df.columns = ["eur_yld_" + col + "_prev_2d" for col in eurostat_euro_yield_df.columns]

    #     return eurostat_euro_yield_df
    
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
        # Melt and pivot operations
        melted = eurostat_euro_yield_df.melt(
            id_vars=eurostat_euro_yield_df.columns[1:4],  # Original id_vars (columns 1-3)
            value_vars=eurostat_euro_yield_df.columns[5:]  # Original value_vars (columns 5+)
        )
        
        # Pivot to wide format
        pivoted = melted.pivot(
            values="value",
            index="variable",
            columns="maturity",
            aggregate_function="first"
        )
        
        # Clean up and date handling
        euro_yield = (
            pivoted
            .rename({"variable": 'datetime'})
            #.with_columns(pl.col('datetime').str.to_datetime())
        )
        
        # Calendar operations
        eurex = mcal.get_calendar('EUREX')
        last_date = euro_yield['datetime'].max()
        
        # Generate new dates using pandas market calendar
        new_dates = eurex.schedule(
            start_date=last_date + pd.DateOffset(days=1),
            end_date=last_date + pd.DateOffset(days=2)
        ).index.tz_convert("UTC")
        
        # Create empty Polars DataFrame for new dates
        new_rows = pl.DataFrame({
            'datetime': pl.from_pandas(new_dates.to_series()).cast(pl.Datetime()),
        }).with_columns(**{col: None for col in euro_yield.columns if col != 'datetime'})
        
        # Combine and shift
        combined = pl.concat([euro_yield, new_rows]).sort('datetime')
        
        # Shift values and slice
        shifted = combined.select(
            pl.all().shift(2).alias("shifted_*")
        ).rename({f"shifted_{col}": col for col in combined.columns})
        
        final_df = shifted.slice(2, len(combined) - 2)
        
        # Rename columns
        return final_df.rename({
            col: f"eur_yld_{col}_prev_2d" 
            for col in final_df.columns 
            if col not in ['datetime']
        })

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

        # results = Parallel(n_jobs=-1, backend="loky")(
        #     delayed(self.read_transform_save)(self.compute_daily_ticker_features,str(ticker_file_path))
        #     for ticker_file_path in self.list_all_files(dir_path) if ticker_file_path.suffix == ".csv"
        # )

        results = []
        for ticker_file_path in self.list_all_files(dir_path):
            if ticker_file_path.suffix == ".csv":
                results.append(self.read_transform_save(self.compute_daily_ticker_features, str(ticker_file_path)))

        return results
    
    def merge_tickers(self, ticker_df_list: List, verbose: Optional[bool] = False)-> pl.DataFrame:

        merged_tickers_df = pl.concat(ticker_df_list)

        if verbose:
            tickers = merged_tickers_df['symbol'].unique().to_list()
            print(f"Merged {len(tickers)} tickers")

        
        return merged_tickers_df


class MacroIndicatorTransformer():
    pass