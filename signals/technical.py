"""
Technical Indicators Module
RSI, MACD, Bollinger Bands, ATR, and other technical signals
"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for trading signals.
    Implements common indicators used in quantitative trading.
    """
    
    def __init__(self, config=None):
        """
        Initialize with configuration.
        
        Args:
            config: SignalConfig object (optional)
        """
        if config:
            self.rsi_period = config.rsi_period
            self.macd_fast = config.macd_fast
            self.macd_slow = config.macd_slow
            self.macd_signal = config.macd_signal
            self.bb_period = config.bb_period
            self.bb_std = config.bb_std
            self.atr_period = config.atr_period
        else:
            # Defaults
            self.rsi_period = 14
            self.macd_fast = 12
            self.macd_slow = 26
            self.macd_signal = 9
            self.bb_period = 20
            self.bb_std = 2.0
            self.atr_period = 14
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data (must have 'open', 'high', 'low', 'close', 'volume')
            
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Price-based indicators
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_sma(df, [10, 20, 50])
        df = self.add_ema(df, [10, 20, 50])
        df = self.add_adx(df)
        
        # Volume indicators
        df = self.add_volume_sma(df)
        df = self.add_obv(df)
        df = self.add_mfi(df)
        df = self.add_chaikin_money_flow(df)
        df = self.add_accumulation_distribution(df)

        return df
    
    def add_rsi(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        period = period or self.rsi_period
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # Use exponential moving average for smoother RSI
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        return df
    
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line)
        Histogram = MACD Line - Signal Line
        """
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD crossover signals
        df['macd_bullish'] = ((df['macd'] > df['macd_signal']) & 
                              (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_bearish'] = ((df['macd'] < df['macd_signal']) & 
                              (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Middle Band = SMA(close, period)
        Upper Band = Middle Band + (std * multiplier)
        Lower Band = Middle Band - (std * multiplier)
        """
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        rolling_std = df['close'].rolling(window=self.bb_period).std()
        
        df['bb_upper'] = df['bb_middle'] + (rolling_std * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * self.bb_std)
        
        # Bollinger Band width (volatility measure)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # %B indicator (position within bands)
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Signals
        df['bb_oversold'] = (df['close'] < df['bb_lower']).astype(int)
        df['bb_overbought'] = (df['close'] > df['bb_upper']).astype(int)
        
        return df
    
    def add_atr(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        
        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = SMA(True Range, period)
        """
        period = period or self.atr_period
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        
        # ATR as percentage of price (normalized volatility)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        return df
    
    def add_sma(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """Add Simple Moving Averages for multiple periods."""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def add_ema(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """Add Exponential Moving Averages for multiple periods."""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).
        
        Measures trend strength (not direction).
        ADX > 25 indicates strong trend.
        """
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate ATR if not already present
        if 'atr' not in df.columns:
            df = self.add_atr(df, period)
        
        # Smoothed +DM and -DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False).mean()
        
        # +DI and -DI
        plus_di = 100 * plus_dm_smooth / df['atr'].replace(0, np.inf)
        minus_di = 100 * minus_dm_smooth / df['atr'].replace(0, np.inf)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
        df['adx'] = dx.ewm(span=period, adjust=False).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Strong trend signal
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        
        return df
    
    def add_volume_sma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add volume moving average and relative volume."""
        df['volume_sma'] = df['volume'].rolling(window=period).mean()
        df['relative_volume'] = df['volume'] / df['volume_sma']
        df['high_volume'] = (df['relative_volume'] > 1.5).astype(int)
        return df
    
    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On-Balance Volume (OBV).
        
        Cumulative volume indicator showing buying/selling pressure.
        """
        obv = []
        obv.append(0)
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['obv'] = obv
        df['obv_sma'] = df['obv'].rolling(window=20).mean()

        return df

    def add_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Money Flow Index (MFI).

        MFI is like RSI but incorporates volume. Ranges from 0-100.
        MFI > 80 = overbought, MFI < 20 = oversold

        Formula:
        1. Typical Price = (High + Low + Close) / 3
        2. Money Flow = Typical Price × Volume
        3. Positive/Negative Money Flow based on price direction
        4. Money Flow Ratio = 14-period Positive MF / 14-period Negative MF
        5. MFI = 100 - (100 / (1 + Money Flow Ratio))
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        # Determine positive vs negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        # Calculate rolling sums
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        # Money Flow Ratio and MFI
        mf_ratio = positive_mf / negative_mf.replace(0, np.inf)
        df['mfi'] = 100 - (100 / (1 + mf_ratio))

        # MFI signals
        df['mfi_oversold'] = (df['mfi'] < 20).astype(int)
        df['mfi_overbought'] = (df['mfi'] > 80).astype(int)

        return df

    def add_chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate Chaikin Money Flow (CMF).

        CMF measures buying/selling pressure over a period.
        Positive = accumulation, Negative = distribution

        Formula:
        1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        2. Money Flow Volume = Multiplier × Volume
        3. CMF = Sum(Money Flow Volume, period) / Sum(Volume, period)

        CMF > 0 = buying pressure, CMF < 0 = selling pressure
        """
        # Money Flow Multiplier
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)  # Handle division by zero when high == low

        # Money Flow Volume
        mf_volume = clv * df['volume']

        # CMF
        df['cmf'] = (
            mf_volume.rolling(window=period).sum() /
            df['volume'].rolling(window=period).sum()
        )

        # CMF signals
        df['cmf_bullish'] = (df['cmf'] > 0.05).astype(int)  # Strong accumulation
        df['cmf_bearish'] = (df['cmf'] < -0.05).astype(int)  # Strong distribution

        return df

    def add_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Accumulation/Distribution Line (ADL).

        ADL is a cumulative indicator that uses volume and price to assess
        whether a stock is being accumulated or distributed.

        Formula:
        1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        2. Money Flow Volume = Multiplier × Volume
        3. ADL = Previous ADL + Money Flow Volume

        Rising ADL = accumulation, Falling ADL = distribution
        """
        # Money Flow Multiplier (same as CMF)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)

        # Money Flow Volume
        mf_volume = clv * df['volume']

        # Cumulative ADL
        df['adl'] = mf_volume.cumsum()

        # ADL trend (20-period rate of change)
        df['adl_roc'] = df['adl'].pct_change(20, fill_method=None)

        return df

    def get_indicator_score(self, row: pd.Series) -> float:
        """
        Calculate a composite technical score from indicators.
        
        Args:
            row: Series with indicator values
            
        Returns:
            Score between -1 (bearish) and 1 (bullish)
        """
        score = 0.0
        weight_sum = 0.0
        
        # RSI score (0.2 weight)
        if 'rsi' in row and pd.notna(row['rsi']):
            if row['rsi'] < 30:
                score += 0.2 * 1.0  # Oversold = bullish
            elif row['rsi'] > 70:
                score += 0.2 * -1.0  # Overbought = bearish
            else:
                score += 0.2 * ((50 - row['rsi']) / 50)  # Neutral zone
            weight_sum += 0.2
        
        # MACD score (0.25 weight)
        if 'macd_histogram' in row and pd.notna(row['macd_histogram']):
            macd_norm = np.clip(row['macd_histogram'] / (abs(row['close']) * 0.02 + 1), -1, 1)
            score += 0.25 * macd_norm
            weight_sum += 0.25
        
        # Bollinger Bands score (0.2 weight)
        if 'bb_pct' in row and pd.notna(row['bb_pct']):
            # Below 0 = oversold (bullish), above 1 = overbought (bearish)
            bb_score = 1 - 2 * np.clip(row['bb_pct'], 0, 1)
            score += 0.2 * bb_score
            weight_sum += 0.2
        
        # Trend score (0.2 weight)
        if 'sma_20' in row and 'sma_50' in row and pd.notna(row['sma_20']) and pd.notna(row['sma_50']):
            if row['close'] > row['sma_20'] > row['sma_50']:
                score += 0.2 * 1.0  # Strong uptrend
            elif row['close'] < row['sma_20'] < row['sma_50']:
                score += 0.2 * -1.0  # Strong downtrend
            else:
                score += 0.2 * 0.0  # No clear trend
            weight_sum += 0.2
        
        # Volume confirmation (0.15 weight)
        if 'high_volume' in row and 'macd_histogram' in row:
            if row['high_volume'] and row['macd_histogram'] > 0:
                score += 0.15 * 0.5  # Bullish with volume
            elif row['high_volume'] and row['macd_histogram'] < 0:
                score += 0.15 * -0.5  # Bearish with volume
            weight_sum += 0.15
        
        # Normalize by weights used
        if weight_sum > 0:
            score = score / weight_sum
        
        return np.clip(score, -1, 1)
