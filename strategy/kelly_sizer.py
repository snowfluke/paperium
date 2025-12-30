"""
Kelly Criterion Position Sizing
Calculates optimal position sizes based on edge and variance.

Kelly Formula: f* = (p * b - q) / b
Where:
- f* = fraction of capital to bet
- p = probability of win
- q = probability of loss (1 - p)
- b = odds received on win (W/L ratio)

We use a fractional Kelly (25-50%) for safety.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class KellyPositionSizer:
    """
    Dynamic position sizing based on Kelly Criterion.

    Adjusts position size based on:
    1. Win probability (ML model confidence)
    2. Win/Loss ratio (expected payoff)
    3. Market regime (volatility adjustment)
    4. Historical edge (recent performance)
    """

    def __init__(
        self,
        base_capital: float = 100_000_000,
        max_position_pct: float = 0.15,  # 15% max per position
        min_position_pct: float = 0.02,  # 2% min per position
        kelly_fraction: float = 0.25,    # Use 25% of full Kelly (conservative)
        min_confidence: float = 0.60,    # Minimum ML confidence to trade
        max_positions: int = 10
    ):
        """
        Initialize Kelly position sizer.

        Args:
            base_capital: Total capital available
            max_position_pct: Maximum % of capital per position
            min_position_pct: Minimum % of capital per position
            kelly_fraction: Fraction of full Kelly to use (0.25-0.5 recommended)
            min_confidence: Minimum model confidence to enter trade
            max_positions: Maximum concurrent positions
        """
        self.base_capital = base_capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.kelly_fraction = kelly_fraction
        self.min_confidence = min_confidence
        self.max_positions = max_positions

        # Track recent performance for adaptive sizing
        self.recent_trades: list = []
        self.max_recent_trades = 50

    def calculate_kelly_fraction(
        self,
        win_prob: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly fraction for position sizing.

        Args:
            win_prob: Probability of winning (0-1)
            win_loss_ratio: Average win / Average loss ratio

        Returns:
            Kelly fraction (0-1)
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        if win_loss_ratio <= 0:
            return 0.0

        # Kelly formula: f* = (p * b - q) / b
        # where b = win/loss ratio, p = win prob, q = 1 - p
        loss_prob = 1 - win_prob
        kelly_full = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio

        # Constrain to [0, 1]
        kelly_full = np.clip(kelly_full, 0, 1)

        # Apply fractional Kelly for safety
        kelly_fractional = kelly_full * self.kelly_fraction

        return kelly_fractional

    def calculate_position_size(
        self,
        ml_confidence: float,
        current_price: float,
        atr: float,
        regime_multiplier: float = 1.0,
        historical_edge: Dict[str, float] = None
    ) -> Tuple[float, int, float]:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            ml_confidence: ML model confidence (0-1)
            current_price: Current stock price
            atr: Average True Range for stop loss calculation
            regime_multiplier: Market regime adjustment (0-1.5)
            historical_edge: Dict with 'win_rate' and 'wl_ratio' from recent trades

        Returns:
            Tuple of (position_value_idr, num_shares, position_pct)
        """
        # 1. Check minimum confidence threshold
        if ml_confidence < self.min_confidence:
            return 0.0, 0, 0.0

        # 2. Get historical edge (or use defaults)
        if historical_edge:
            win_rate = historical_edge.get('win_rate', 0.70)
            wl_ratio = historical_edge.get('wl_ratio', 2.0)
        else:
            # Conservative defaults if no history
            win_rate = 0.70
            wl_ratio = 2.0

        # 3. Adjust win rate based on ML confidence
        # Higher confidence → increase win rate estimate
        adjusted_win_rate = win_rate + (ml_confidence - 0.70) * 0.3  # Scale confidence impact
        adjusted_win_rate = np.clip(adjusted_win_rate, 0.50, 0.95)

        # 4. Calculate Kelly fraction
        kelly_frac = self.calculate_kelly_fraction(adjusted_win_rate, wl_ratio)

        # 5. Apply regime multiplier
        kelly_frac *= regime_multiplier

        # 6. Constrain to min/max position size
        position_pct = np.clip(kelly_frac, self.min_position_pct, self.max_position_pct)

        # 7. Calculate position value in IDR
        position_value = self.base_capital * position_pct

        # 8. Calculate number of shares (round down to nearest 100-lot for IDX)
        shares_raw = position_value / current_price
        shares_rounded = int(shares_raw // 100) * 100  # Round to nearest 100

        # Recalculate actual position value based on rounded shares
        actual_position_value = shares_rounded * current_price
        actual_position_pct = actual_position_value / self.base_capital

        logger.debug(f"Kelly Sizing: Confidence={ml_confidence:.2f}, "
                    f"WR={adjusted_win_rate:.2f}, WL={wl_ratio:.2f}, "
                    f"Kelly={kelly_frac:.3f}, Regime={regime_multiplier:.2f}, "
                    f"Position={actual_position_pct:.2%}")

        return actual_position_value, shares_rounded, actual_position_pct

    def update_performance(self, trade_result: Dict):
        """
        Update recent performance tracking for adaptive sizing.

        Args:
            trade_result: Dict with 'win' (bool) and 'pnl_pct' (float)
        """
        self.recent_trades.append(trade_result)

        # Keep only recent trades
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades = self.recent_trades[-self.max_recent_trades:]

    def get_recent_edge(self) -> Dict[str, float]:
        """
        Calculate recent historical edge from tracked trades.

        Returns:
            Dict with 'win_rate', 'wl_ratio', 'total_trades'
        """
        if not self.recent_trades:
            return {'win_rate': 0.70, 'wl_ratio': 2.0, 'total_trades': 0}

        wins = [t for t in self.recent_trades if t['win']]
        losses = [t for t in self.recent_trades if not t['win']]

        win_rate = len(wins) / len(self.recent_trades) if self.recent_trades else 0.70

        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0.05
        avg_loss = abs(np.mean([t['pnl_pct'] for t in losses])) if losses else 0.03
        wl_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0

        return {
            'win_rate': win_rate,
            'wl_ratio': wl_ratio,
            'total_trades': len(self.recent_trades)
        }

    def should_reduce_exposure(self, current_positions: int, recent_losses: int) -> bool:
        """
        Check if position sizing should be reduced due to drawdown.

        Args:
            current_positions: Number of current open positions
            recent_losses: Number of consecutive recent losses

        Returns:
            True if should reduce exposure
        """
        # Reduce if:
        # 1. Near max positions
        # 2. Experiencing consecutive losses (3+)
        if current_positions >= self.max_positions * 0.8 and recent_losses >= 3:
            return True

        # Severe drawdown (5+ consecutive losses)
        if recent_losses >= 5:
            return True

        return False

    def get_max_position_value(self) -> float:
        """Get maximum position value in IDR."""
        return self.base_capital * self.max_position_pct

    def get_min_position_value(self) -> float:
        """Get minimum position value in IDR."""
        return self.base_capital * self.min_position_pct
