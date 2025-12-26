"""
Exit Manager Module
ATR-based dynamic stops, trailing stops, and take profit targets
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExitLevels:
    """Container for exit price levels."""
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop_distance: float
    max_hold_days: int
    atr: float
    risk_amount: float  # Dollar risk per share
    reward_ratio: float  # Risk:Reward ratio


class ExitManager:
    """
    Manages exit strategies for positions.
    Implements Triple Barrier Method with ATR-based dynamic levels.
    """
    
    def __init__(self, config=None):
        """
        Initialize exit manager.
        
        Args:
            config: ExitConfig object (optional)
        """
        if config:
            self.stop_loss_atr = config.stop_loss_atr_mult
            self.take_profit_atr = config.take_profit_atr_mult
            self.trailing_stop_atr = config.trailing_stop_atr_mult
            self.max_hold_days = config.max_holding_days
            self.max_loss_pct = config.max_loss_pct
            self.min_profit_pct = config.min_profit_pct
        else:
            self.stop_loss_atr = 2.0
            self.take_profit_atr = 3.0
            self.trailing_stop_atr = 1.5
            self.max_hold_days = 5
            self.max_loss_pct = 0.08
            self.min_profit_pct = 0.02
    
    def calculate_levels(
        self, 
        entry_price: float, 
        atr: float,
        direction: str = 'LONG',
        mode: str = 'BASELINE'
    ) -> ExitLevels:
        """
        Calculate exit levels based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Average True Range value
            direction: 'LONG' or 'SHORT'
            
        Returns:
            ExitLevels with calculated prices
        """
        # Adaptive multipliers based on strategy mode
        sl_mult = self.stop_loss_atr
        tp_mult = self.take_profit_atr
        
        if mode == 'EXPLOSIVE':
            sl_mult *= 0.8  # Tighter stop for higher conviction entries
            tp_mult *= 1.5  # Aim for larger breakout move
            
        if direction == 'LONG':
            stop_loss = entry_price - (sl_mult * atr)
            take_profit = entry_price + (tp_mult * atr)
        else:  # SHORT
            stop_loss = entry_price + (sl_mult * atr)
            take_profit = entry_price - (tp_mult * atr)
        
        # Apply maximum loss limit
        max_stop = entry_price * (1 - self.max_loss_pct) if direction == 'LONG' else entry_price * (1 + self.max_loss_pct)
        
        if direction == 'LONG':
            stop_loss = max(stop_loss, max_stop)
        else:
            stop_loss = min(stop_loss, max_stop)
        
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(take_profit - entry_price)
        reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return ExitLevels(
            entry_price=entry_price,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            trailing_stop_distance=round(self.trailing_stop_atr * atr, 2),
            max_hold_days=self.max_hold_days,
            atr=round(atr, 2),
            risk_amount=round(risk_amount, 2),
            reward_ratio=round(reward_ratio, 2)
        )
    
    def check_exit_conditions(
        self,
        current_price: float,
        high_price: float,
        exit_levels: ExitLevels,
        days_held: int,
        direction: str = 'LONG'
    ) -> Dict:
        """
        Check if any exit condition is triggered.
        
        Args:
            current_price: Current market price
            high_price: Highest price since entry (for trailing stop)
            exit_levels: ExitLevels object
            days_held: Number of days position has been held
            direction: 'LONG' or 'SHORT'
            
        Returns:
            Dictionary with exit signal and reason
        """
        result = {
            'should_exit': False,
            'exit_reason': None,
            'exit_type': None,
            'exit_price': current_price
        }
        
        # Check stop loss
        if direction == 'LONG':
            if current_price <= exit_levels.stop_loss:
                result['should_exit'] = True
                result['exit_reason'] = 'STOP_LOSS'
                result['exit_type'] = 'loss'
                return result
        else:
            if current_price >= exit_levels.stop_loss:
                result['should_exit'] = True
                result['exit_reason'] = 'STOP_LOSS'
                result['exit_type'] = 'loss'
                return result
        
        # Check take profit
        if direction == 'LONG':
            if current_price >= exit_levels.take_profit:
                result['should_exit'] = True
                result['exit_reason'] = 'TAKE_PROFIT'
                result['exit_type'] = 'profit'
                return result
        else:
            if current_price <= exit_levels.take_profit:
                result['should_exit'] = True
                result['exit_reason'] = 'TAKE_PROFIT'
                result['exit_type'] = 'profit'
                return result
        
        # Check trailing stop
        trailing_stop = self._calculate_trailing_stop(
            high_price, 
            exit_levels.trailing_stop_distance,
            direction
        )
        
        if direction == 'LONG':
            if current_price <= trailing_stop and high_price > exit_levels.entry_price:
                result['should_exit'] = True
                result['exit_reason'] = 'TRAILING_STOP'
                result['exit_type'] = 'profit' if current_price > exit_levels.entry_price else 'loss'
                return result
        else:
            if current_price >= trailing_stop and high_price < exit_levels.entry_price:
                result['should_exit'] = True
                result['exit_reason'] = 'TRAILING_STOP'
                result['exit_type'] = 'profit' if current_price < exit_levels.entry_price else 'loss'
                return result
        
        # Check time-based exit
        if days_held >= exit_levels.max_hold_days:
            result['should_exit'] = True
            result['exit_reason'] = 'TIME_STOP'
            pnl = (current_price - exit_levels.entry_price) / exit_levels.entry_price
            result['exit_type'] = 'profit' if pnl > 0 else 'loss'
            return result
        
        return result
    
    def _calculate_trailing_stop(
        self, 
        high_price: float, 
        trailing_distance: float,
        direction: str
    ) -> float:
        """Calculate current trailing stop price."""
        if direction == 'LONG':
            return high_price - trailing_distance
        else:
            return high_price + trailing_distance
    
    def update_trailing_stop(
        self,
        current_high: float,
        previous_trailing: float,
        trailing_distance: float,
        direction: str = 'LONG'
    ) -> float:
        """
        Update trailing stop if price has moved favorably.
        
        Returns new trailing stop level (only moves in profitable direction).
        """
        new_trailing = self._calculate_trailing_stop(
            current_high, 
            trailing_distance, 
            direction
        )
        
        if direction == 'LONG':
            return max(new_trailing, previous_trailing)
        else:
            return min(new_trailing, previous_trailing)
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float,
        direction: str = 'LONG'
    ) -> Dict:
        """
        Calculate P&L for a closed position.
        
        Returns:
            Dictionary with P&L details
        """
        if direction == 'LONG':
            pnl = (exit_price - entry_price) * position_size
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) * position_size
            pnl_pct = (entry_price - exit_price) / entry_price
        
        return {
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct * 100, 2),
            'is_winner': pnl > 0
        }


class TradeTracker:
    """
    Tracks open positions and monitors exit conditions.
    """
    
    def __init__(self, exit_manager: ExitManager):
        self.exit_manager = exit_manager
        self.open_positions: Dict[str, Dict] = {}
        self.closed_positions: List[Dict] = []
    
    def open_position(
        self,
        ticker: str,
        entry_price: float,
        atr: float,
        position_size: float,
        direction: str = 'LONG',
        entry_date: Optional[datetime] = None
    ):
        """Open a new position."""
        exit_levels = self.exit_manager.calculate_levels(entry_price, atr, direction)
        
        self.open_positions[ticker] = {
            'ticker': ticker,
            'entry_price': entry_price,
            'entry_date': entry_date or datetime.now(),
            'position_size': position_size,
            'direction': direction,
            'exit_levels': exit_levels,
            'high_since_entry': entry_price,
            'low_since_entry': entry_price
        }
        
        logger.info(f"Opened {direction} position: {ticker} @ {entry_price}, SL: {exit_levels.stop_loss}, TP: {exit_levels.take_profit}")
    
    def update_position(
        self,
        ticker: str,
        current_price: float,
        current_high: float,
        current_low: float,
        current_date: Optional[datetime] = None
    ) -> Optional[Dict]:
        """
        Update position with current prices and check exits.
        
        Returns exit signal if triggered, None otherwise.
        """
        if ticker not in self.open_positions:
            return None
        
        pos = self.open_positions[ticker]
        
        # Update high/low since entry
        pos['high_since_entry'] = max(pos['high_since_entry'], current_high)
        pos['low_since_entry'] = min(pos['low_since_entry'], current_low)
        
        # Calculate days held
        entry_date = pos['entry_date']
        current = current_date or datetime.now()
        days_held = (current - entry_date).days
        
        # Check exit conditions
        exit_signal = self.exit_manager.check_exit_conditions(
            current_price=current_price,
            high_price=pos['high_since_entry'] if pos['direction'] == 'LONG' else pos['low_since_entry'],
            exit_levels=pos['exit_levels'],
            days_held=days_held,
            direction=pos['direction']
        )
        
        if exit_signal['should_exit']:
            self.close_position(ticker, current_price, exit_signal['exit_reason'])
            return exit_signal
        
        return None
    
    def close_position(
        self,
        ticker: str,
        exit_price: float,
        exit_reason: str
    ) -> Dict:
        """Close a position and record results."""
        if ticker not in self.open_positions:
            return {}
        
        pos = self.open_positions.pop(ticker)
        
        pnl = self.exit_manager.calculate_pnl(
            pos['entry_price'],
            exit_price,
            pos['position_size'],
            pos['direction']
        )
        
        closed = {
            **pos,
            'exit_price': exit_price,
            'exit_date': datetime.now(),
            'exit_reason': exit_reason,
            **pnl
        }
        
        self.closed_positions.append(closed)
        
        logger.info(f"Closed {ticker}: {exit_reason}, P&L: {pnl['pnl_pct']}%")
        
        return closed
    
    def get_statistics(self) -> Dict:
        """Get trading statistics."""
        if not self.closed_positions:
            return {}
        
        df = pd.DataFrame(self.closed_positions)
        
        winners = df[df['is_winner']]
        losers = df[~df['is_winner']]
        
        return {
            'total_trades': len(df),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(df) if len(df) > 0 else 0,
            'avg_win': winners['pnl_pct'].mean() if len(winners) > 0 else 0,
            'avg_loss': losers['pnl_pct'].mean() if len(losers) > 0 else 0,
            'profit_factor': abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else float('inf'),
            'total_pnl': df['pnl'].sum()
        }
