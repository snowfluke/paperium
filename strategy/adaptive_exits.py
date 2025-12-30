"""
Adaptive Stop Loss / Take Profit System

Dynamically adjusts exit levels based on:
1. Market volatility regime
2. Supply/Demand zone proximity
3. Position conviction (ML confidence)
4. Time decay

This replaces static SL/TP with intelligent, context-aware exits.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from signals.regime_detector import MarketRegime
import logging

logger = logging.getLogger(__name__)


class AdaptiveExitManager:
    """
    Intelligent exit management with regime-aware stops and targets.

    Exit Strategy:
    - CRASH: No trades
    - HIGH_VOL: Tighter stops (1.5x ATR), wider targets (5x ATR)
    - NORMAL: Balanced (2x ATR stop, 4x ATR target)
    - LOW_VOL: Wider stops (2.5x ATR), tighter targets (3x ATR)

    Additional Features:
    - Partial profit taking at 50% of target
    - Trailing stops after reaching 1.5x ATR profit
    - Time decay (tighten stops after 3 days)
    - S/D zone-aware exits (tighten near supply)
    """

    def __init__(
        self,
        max_hold_days: int = 5,
        partial_take_pct: float = 0.50,  # Take 50% at partial target
        trailing_activation: float = 1.5,  # Activate trailing after 1.5x ATR
        trailing_distance: float = 1.0    # Trail at 1x ATR below high
    ):
        """
        Initialize adaptive exit manager.

        Args:
            max_hold_days: Maximum days to hold position
            partial_take_pct: Percentage to take at partial target
            trailing_activation: ATR multiplier to activate trailing stop
            trailing_distance: ATR multiplier for trailing stop distance
        """
        self.max_hold_days = max_hold_days
        self.partial_take_pct = partial_take_pct
        self.trailing_activation = trailing_activation
        self.trailing_distance = trailing_distance

        # Regime-specific parameters
        self.regime_params = {
            MarketRegime.CRASH: {
                'sl_mult': 0.0,  # No trades
                'tp_mult': 0.0,
                'enabled': False
            },
            MarketRegime.HIGH_VOL: {
                'sl_mult': 1.5,  # Tighter stop in volatile market
                'tp_mult': 5.0,  # Wider target to capture big moves
                'enabled': True
            },
            MarketRegime.NORMAL: {
                'sl_mult': 2.0,
                'tp_mult': 4.0,
                'enabled': True
            },
            MarketRegime.LOW_VOL: {
                'sl_mult': 2.5,  # Wider stop in calm market (avoid noise)
                'tp_mult': 3.0,  # Tighter target (limited upside)
                'enabled': True
            }
        }

    def get_exit_levels(
        self,
        entry_price: float,
        atr: float,
        regime: MarketRegime,
        ml_confidence: float = 0.7,
        sd_supply_distance: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate adaptive stop loss and take profit levels.

        Args:
            entry_price: Entry price
            atr: Average True Range
            regime: Current market regime
            ml_confidence: ML model confidence (0-1)
            sd_supply_distance: Distance to nearest supply zone (%)

        Returns:
            Dict with 'stop_loss', 'take_profit', 'partial_target'
        """
        params = self.regime_params.get(regime, self.regime_params[MarketRegime.NORMAL])

        if not params['enabled']:
            return {
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'partial_target': 0.0
            }

        # Base levels from regime
        sl_distance = atr * params['sl_mult']
        tp_distance = atr * params['tp_mult']

        # Confidence adjustment (higher confidence = wider stops, targets)
        confidence_factor = 0.8 + (ml_confidence - 0.5) * 0.4  # 0.8-1.2 range
        sl_distance *= confidence_factor
        tp_distance *= confidence_factor

        # S/D zone adjustment (tighten TP if near supply)
        if sd_supply_distance is not None and sd_supply_distance < 0.05:  # Within 5%
            # Supply overhead - reduce target by 20%
            tp_distance *= 0.8
            logger.debug(f"TP reduced due to nearby supply zone (distance={sd_supply_distance:.2%})")

        # Calculate levels
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance
        partial_target = entry_price + (tp_distance * 0.5)  # 50% of full target

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'partial_target': partial_target,
            'sl_distance_pct': sl_distance / entry_price,
            'tp_distance_pct': tp_distance / entry_price
        }

    def check_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr: float
    ) -> Optional[float]:
        """
        Calculate trailing stop level if activated.

        Args:
            entry_price: Entry price
            current_price: Current price
            highest_price: Highest price since entry
            atr: Average True Range

        Returns:
            Trailing stop price or None if not activated
        """
        # Check if profit reached activation threshold
        profit = current_price - entry_price
        activation_threshold = atr * self.trailing_activation

        if profit < activation_threshold:
            return None  # Not activated yet

        # Calculate trailing stop (trail from highest price)
        trailing_distance = atr * self.trailing_distance
        trailing_stop = highest_price - trailing_distance

        # Ensure trailing stop is above entry (never trail below breakeven)
        trailing_stop = max(trailing_stop, entry_price)

        return trailing_stop

    def check_time_decay(
        self,
        days_held: int,
        current_sl: float,
        current_tp: float,
        entry_price: float
    ) -> Tuple[float, float]:
        """
        Apply time decay to exit levels.

        After 3 days, start tightening stops to lock in profits.

        Args:
            days_held: Number of days position has been held
            current_sl: Current stop loss
            current_tp: Current take profit
            entry_price: Entry price

        Returns:
            Tuple of (adjusted_sl, adjusted_tp)
        """
        if days_held <= 2:
            return current_sl, current_tp  # No decay yet

        # Time decay factor (increases with days held)
        decay_days = days_held - 2
        decay_factor = min(0.5, decay_days * 0.15)  # Max 50% tightening

        # Tighten stop loss (move towards entry)
        sl_distance = entry_price - current_sl
        new_sl_distance = sl_distance * (1 - decay_factor)
        adjusted_sl = entry_price - new_sl_distance

        # Slightly reduce target (encourage faster exit)
        tp_distance = current_tp - entry_price
        new_tp_distance = tp_distance * (1 - decay_factor * 0.5)  # Less aggressive on TP
        adjusted_tp = entry_price + new_tp_distance

        logger.debug(f"Time decay applied (days={days_held}): SL {current_sl:.0f} → {adjusted_sl:.0f}, "
                    f"TP {current_tp:.0f} → {adjusted_tp:.0f}")

        return adjusted_sl, adjusted_tp

    def should_exit(
        self,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        take_profit: float,
        days_held: int,
        highest_price: float,
        atr: float
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Args:
            entry_price: Entry price
            current_price: Current price
            stop_loss: Stop loss level
            take_profit: Take profit level
            days_held: Days held
            highest_price: Highest price since entry
            atr: Average True Range

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        # 1. Check stop loss
        if current_price <= stop_loss:
            return True, "STOP_LOSS"

        # 2. Check take profit
        if current_price >= take_profit:
            return True, "TAKE_PROFIT"

        # 3. Check trailing stop
        trailing_stop = self.check_trailing_stop(entry_price, current_price, highest_price, atr)
        if trailing_stop is not None and current_price <= trailing_stop:
            return True, "TRAILING_STOP"

        # 4. Check max hold time
        if days_held >= self.max_hold_days:
            return True, "MAX_HOLD_DAYS"

        return False, "HOLD"

    def calculate_partial_exit_size(
        self,
        current_price: float,
        partial_target: float,
        total_shares: int
    ) -> int:
        """
        Calculate number of shares to sell at partial target.

        Args:
            current_price: Current price
            partial_target: Partial target price
            total_shares: Total shares in position

        Returns:
            Number of shares to sell (rounded to 100-lot)
        """
        if current_price < partial_target:
            return 0

        # Sell 50% at partial target
        partial_shares = int(total_shares * self.partial_take_pct)

        # Round to 100-lot
        partial_shares = (partial_shares // 100) * 100

        return partial_shares

    def get_exit_summary(
        self,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        shares: int,
        days_held: int
    ) -> Dict:
        """
        Generate exit summary statistics.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            exit_reason: Reason for exit
            shares: Number of shares
            days_held: Days held

        Returns:
            Exit summary dictionary
        """
        pnl = (exit_price - entry_price) * shares
        pnl_pct = (exit_price / entry_price - 1) * 100

        return {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'shares': shares,
            'days_held': days_held,
            'exit_reason': exit_reason,
            'is_win': pnl > 0
        }
