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
    CRASH = "CRASH"  # IHSG is crashing - avoid all trades


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
    
    def detect_ihsg_crash(
        self,
        ihsg_prices: pd.Series,
        current_date: Optional[pd.Timestamp] = None,
        crash_threshold: float = -0.05,
        crash_window: int = 5,
        drawdown_threshold: float = -0.10
    ) -> bool:
        """
        Detect if IHSG (Indonesian market) is crashing.

        Hypothesis: Trades fail when the broad market is crashing.
        This is a HARD FILTER - returns True if market is crashing (avoid all trades).

        Args:
            ihsg_prices: Series of IHSG closing prices
            current_date: Date to check (defaults to latest)
            crash_threshold: Return threshold for single-day crash (default -5%)
            crash_window: Number of recent days to check for sustained decline
            drawdown_threshold: Drawdown from recent high (default -10%)

        Returns:
            True if market is crashing, False otherwise
        """
        if len(ihsg_prices) < crash_window + 20:
            return False  # Insufficient data

        if current_date is None:
            current_date = ihsg_prices.index[-1]

        # Get recent prices
        try:
            recent_prices = ihsg_prices.loc[:current_date].tail(crash_window + 20)
        except (KeyError, TypeError):
            recent_prices = ihsg_prices.tail(crash_window + 20)

        if len(recent_prices) < crash_window:
            return False

        # Condition 1: Single-day crash (>5% drop)
        daily_returns = recent_prices.pct_change()
        latest_return = daily_returns.iloc[-1]
        if latest_return <= crash_threshold:
            logger.debug(f"IHSG CRASH: Single-day drop of {latest_return:.2%}")
            return True

        # Condition 2: Sustained decline (negative returns over crash_window days)
        window_return = recent_prices.iloc[-1] / recent_prices.iloc[-crash_window] - 1
        if window_return <= crash_threshold:
            logger.debug(f"IHSG CRASH: {crash_window}-day decline of {window_return:.2%}")
            return True

        # Condition 3: Drawdown from recent high (20-day high)
        recent_high = recent_prices.tail(20).max()
        current_price = recent_prices.iloc[-1]
        drawdown = (current_price - recent_high) / recent_high
        if drawdown <= drawdown_threshold:
            logger.debug(f"IHSG CRASH: {drawdown:.2%} drawdown from 20-day high")
            return True

        return False

    def detect_regime_with_crash_filter(
        self,
        prices: pd.Series,
        ihsg_prices: Optional[pd.Series] = None,
        current_date: Optional[pd.Timestamp] = None
    ) -> MarketRegime:
        """
        Detect market regime with IHSG crash filter.

        Args:
            prices: Series of stock closing prices
            ihsg_prices: Series of IHSG index prices (optional)
            current_date: Date to classify (defaults to latest)

        Returns:
            MarketRegime enum (CRASH takes priority over other regimes)
        """
        # First check for IHSG crash
        if ihsg_prices is not None:
            is_crashing = self.detect_ihsg_crash(ihsg_prices, current_date)
            if is_crashing:
                return MarketRegime.CRASH

        # If no crash, use normal volatility-based regime detection
        return self.detect_regime(prices, current_date)

    def get_position_multiplier(self, regime: MarketRegime) -> float:
        """
        Get position size multiplier based on regime.

        Args:
            regime: Current market regime

        Returns:
            Multiplier to apply to position size
            - CRASH: 0.0 (no trades)
            - HIGH_VOL: 0.7 (defensive)
            - NORMAL: 1.0 (baseline)
            - LOW_VOL: 1.2 (slightly aggressive)
        """
        if regime == MarketRegime.CRASH:
            return 0.0  # No trades during market crash
        elif regime == MarketRegime.HIGH_VOL:
            return 0.7  # Reduce position size in high volatility
        elif regime == MarketRegime.LOW_VOL:
            return 1.2  # Slightly increase in low volatility
        else:
            return 1.0
