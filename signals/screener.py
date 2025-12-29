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
        self.min_price = 50  # No penny stocks below 50
        self.min_volume = 1_000_000  # Minimum liquidity 

        
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
        1. Liquidity: Average volume > threshold
        2. Trend: Price > EMA 200 (Long term uptrend)
        3. Momentum: RSI > 50 (Positive momentum)
        4. Volatility: ATR > 1% of price (Enough movement to trade)
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

            # 3. Trend Check (Price > EMA 200 is visually powerful)
            ema200 = df['close'].ewm(span=200).mean().iloc[-1]
            if latest['close'] < ema200:
                return False
                
            # 4. Momentum Check (RSI > 50)
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
                
            # 5. Volatility Potential (ATR > 1% of price)
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
                
            # 6. Circuit Breaker Check (ARA/ARB)
            if not self._check_circuit_breaker(latest['close'], df['close'].shift(1).iloc[-1]):
                return False

            return True
            
        except Exception as e:
            logger.debug(f"Screening failed for {ticker}: {e}")
            return False

    def _check_circuit_breaker(self, current_price: float, prev_close: float) -> bool:
        """
        Check if stock is locked at Auto Rejection limits (ARA/ARB).
        
        IDX Regulations (2025):
        - ARB (Lower): Flat 15% (Effective Apr 2025)
        - ARA (Upper):
            - Price < 200: 35%
            - Price 200-5000: 25%
            - Price > 5000: 20%
            
        Returns:
            True if SAFE (not locked), False if LOCKED (ARA/ARB)
        """
        if prev_close == 0:
            return True
            
        pct_change = (current_price - prev_close) / prev_close
        
        # 1. Determine ARA Limit based on price tier (using prev_close as basis)
        if prev_close < 200:
            ara_cap = 0.35
        elif prev_close <= 5000:
            ara_cap = 0.25
        else:
            ara_cap = 0.20
            
        # 2. Determine ARB Limit 
        # Note: New regulation effective Apr 2025 sets uniform 15% ARB
        # Safe assumption for late 2025 context
        arb_cap = 0.15
        
        
        # Check buffer (within 0.5% of limit is considered "locked/risky")
        # ARA Check: Don't buy if locked up (cannot enter)
        if pct_change >= (ara_cap - 0.005):
            return False
            
        # ARB Check: Don't buy if locked down (falling knife)
        if pct_change <= -(arb_cap - 0.005):
            return False
            
        return True

