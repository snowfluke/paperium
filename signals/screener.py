"""
Market Screener Module
Filters stocks for high potential candidates before detailed analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class Screener:
    """
    Screens stocks for basic technical criteria to identify high-potential candidates.
    Filters out noise and low-probability setups before heavy ML processing.
    """
    
    def __init__(self, config=None):
        self.min_price = 200  # Increased from 50 to filter penny stocks
        self.min_volume = 2_000_000  # Increased from 1M to 2M
        self.min_value = 5_000_000_000 # 5 Billion IDR daily transaction value

        
    def screen_stocks(self, data_map: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Screen all available stocks for candidates.
        
        Args:
            data_map: Dictionary mapping ticker to OHLCV DataFrame
            
        Returns:
            List of tickers that passed the screen
        """
        passed = []
        
        for ticker, df in data_map.items():
            if self._check_criteria(df, ticker):
                passed.append(ticker)
                
        return passed
    
    def _check_criteria(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Check if a single stock meets screening criteria.
        
        Criteria:
        1. Liquidity: Avg Vol > 2M AND Avg Value > 2B IDR
        2. Price: > 200
        3. Trend: Price > EMA 200 (Long term uptrend)
        4. Momentum: RSI > 50 (Positive momentum)
        5. Volatility: ATR > 1% of price (Enough movement to trade)
        """
        if df.empty or len(df) < 200:
            return False
            
        try:
            # Get latest data
            latest = df.iloc[-1]
            
            # 1. Price check
            if latest['close'] < self.min_price:
                return False
                
            # 2. Liquidity check (20-day avg volume)
            avg_vol = df['volume'].rolling(20).mean().iloc[-1]
            if avg_vol < self.min_volume:
                return False

            # 3. Transaction Value Check (Approximate)
            # Volume * Price > 2 Billion
            avg_val = avg_vol * latest['close']
            if avg_val < self.min_value:
                return False
            
            # 4. Trend Check (Price > EMA 200 is visually powerful)
            ema200 = df['close'].ewm(span=200).mean().iloc[-1]
            if latest['close'] < ema200:
                return False
                
            # 5. Momentum Check (RSI > 50)
            # Calculate simple RSI if not present
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
            else:
                current_rsi = latest['rsi']
                
            if current_rsi < 50:
                return False
                
            # 6. Volatility Potential (ATR > 1% of price)
            # We want stocks that move, not dead ones
            if 'atr' not in df.columns:
                high_low = df['high'] - df['low']
                high_close = pd.Series(np.abs(df['high'] - df['close'].shift()), index=df.index)
                low_close = pd.Series(np.abs(df['low'] - df['close'].shift()), index=df.index)
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
            else:
                atr = latest['atr']
                
            atr_pct = (atr / latest['close']) * 100
            if atr_pct < 1.0: # Minimum 1% daily range
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Screening failed for {ticker}: {e}")
            return False

