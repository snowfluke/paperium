"""
Market Regime Detector
Classifies market conditions based on volatility of the IHSG index (^JKSE).
"""
import pandas as pd
import numpy as np
from typing import Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"


class RegimeDetector:
    """
    Detects market regime based on the IHSG index volatility.
    
    Uses a rolling volatility percentile approach:
    - HIGH_VOL: Volatility > 80th percentile (defensive mode)
    - LOW_VOL: Volatility < 20th percentile (aggressive mode)
    - NORMAL: Everything else
    """
    
    def __init__(
        self, 
        lookback_days: int = 60,
        high_vol_percentile: float = 0.80,
        low_vol_percentile: float = 0.20,
        volatility_window: int = 20
    ):
        """
        Initialize regime detector.
        
        Args:
            lookback_days: Days to calculate volatility percentile over
            high_vol_percentile: Percentile above which is HIGH_VOL
            low_vol_percentile: Percentile below which is LOW_VOL
            volatility_window: Rolling window for volatility calculation
        """
        self.lookback_days = lookback_days
        self.high_vol_percentile = high_vol_percentile
        self.low_vol_percentile = low_vol_percentile
        self.volatility_window = volatility_window
    
    def calculate_volatility(self, prices: pd.Series) -> pd.Series:
        """
        Calculate annualized rolling volatility.
        
        Args:
            prices: Series of closing prices
            
        Returns:
            Series of annualized volatility
        """
        safe_ratio = (prices / prices.shift(1)).replace([0, np.inf, -np.inf], np.nan).fillna(1.0)
        log_returns = pd.Series(np.log(safe_ratio.abs()), index=safe_ratio.index)

        volatility = log_returns.rolling(self.volatility_window).std() * np.sqrt(252)
        return volatility
    
    def detect_regime(self, prices: pd.Series, current_date: Optional[pd.Timestamp] = None) -> MarketRegime:
        """
        Detect the current market regime.
        
        Args:
            prices: Series of closing prices (must have DatetimeIndex)
            current_date: Date to classify (defaults to latest)
            
        Returns:
            MarketRegime enum value
        """
        if len(prices) < self.lookback_days + self.volatility_window:
            logger.warning("Insufficient data for regime detection, defaulting to NORMAL")
            return MarketRegime.NORMAL
        
        volatility = self.calculate_volatility(prices)

        # Determine the date to use for regime detection
        if current_date is None:
            current_date = prices.index[-1]

        # Ensure current_date is in volatility index
        if current_date not in volatility.index:
            current_date = volatility.index[-1]

        # At this point, current_date is guaranteed to be a valid index
        # Assert to help type checker understand current_date is not None
        assert current_date is not None, "current_date should not be None after initialization"

        # Get current volatility value
        if current_date in volatility.index:
            # Use .get() to avoid index type issues
            current_vol_value = volatility.get(current_date)
            current_vol = current_vol_value if current_vol_value is not None else volatility.iloc[-1]
        else:
            # Fallback to last value
            current_vol = volatility.iloc[-1]

        # Get historical volatility for percentile calculation
        try:
            historical_vol = volatility.loc[:current_date].tail(self.lookback_days)
        except (KeyError, TypeError):
            historical_vol = volatility.tail(self.lookback_days)
        
        if len(historical_vol) < 10:
            return MarketRegime.NORMAL
        
        # Calculate percentile rank
        percentile_rank = (historical_vol < current_vol).mean()
        
        if percentile_rank >= self.high_vol_percentile:
            return MarketRegime.HIGH_VOL
        elif percentile_rank <= self.low_vol_percentile:
            return MarketRegime.LOW_VOL
        else:
            return MarketRegime.NORMAL
    
    def get_regime_series(self, prices: pd.Series) -> pd.Series:
        """
        Get regime classification for each date in the price series.
        
        Args:
            prices: Series of closing prices
            
        Returns:
            Series of MarketRegime values
        """
        volatility = self.calculate_volatility(prices)
        regimes = []
        
        for i, _ in enumerate(volatility.index):
            if i < self.lookback_days + self.volatility_window:
                regimes.append(MarketRegime.NORMAL)
                continue
            
            current_vol = volatility.iloc[i]
            historical_vol = volatility.iloc[max(0, i - self.lookback_days):i]
            
            if len(historical_vol) < 10:
                regimes.append(MarketRegime.NORMAL)
                continue
            
            percentile_rank = (historical_vol < current_vol).mean()
            
            if percentile_rank >= self.high_vol_percentile:
                regimes.append(MarketRegime.HIGH_VOL)
            elif percentile_rank <= self.low_vol_percentile:
                regimes.append(MarketRegime.LOW_VOL)
            else:
                regimes.append(MarketRegime.NORMAL)
        
        return pd.Series(regimes, index=volatility.index)
    
    def get_position_multiplier(self, regime: MarketRegime) -> float:
        """
        Get position size multiplier based on regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Multiplier to apply to position size (0.7 for HIGH_VOL, 1.0 for NORMAL, 1.2 for LOW_VOL)
        """
        if regime == MarketRegime.HIGH_VOL:
            return 0.7  # Reduce position size in high volatility
        elif regime == MarketRegime.LOW_VOL:
            return 1.2  # Slightly increase in low volatility
        else:
            return 1.0
