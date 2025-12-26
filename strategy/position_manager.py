"""
Position Manager Module
Tracks open positions, pending orders, and position states
"""
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import os

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class PositionStatus(Enum):
    PENDING = "PENDING"          # Limit order not yet filled
    OPEN = "OPEN"                # Position is active
    FLOATING_PROFIT = "FLOATING_PROFIT"
    FLOATING_LOSS = "FLOATING_LOSS"
    HIT_TAKE_PROFIT = "HIT_TAKE_PROFIT"
    HIT_STOP_LOSS = "HIT_STOP_LOSS"
    ENTRY_NOT_HIT = "ENTRY_NOT_HIT"  # Limit order expired
    CLOSED = "CLOSED"


@dataclass
class Position:
    """Represents a trading position."""
    ticker: str
    order_type: str  # MARKET or LIMIT
    entry_price: float
    limit_price: Optional[float]  # For limit orders
    stop_loss: float
    take_profit: float
    shares: int
    position_value: float
    signal_score: float
    strategy_mode: str  # BASELINE or EXPLOSIVE
    created_date: str
    status: str
    filled_date: Optional[str] = None
    filled_price: Optional[float] = None
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class PositionManager:
    """
    Manages trading positions with persistence.
    Tracks pending orders, open positions, and closed trades.
    """
    
    def __init__(self, db_path: str = "data/positions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize positions database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    limit_price REAL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    position_value REAL NOT NULL,
                    signal_score REAL,
                    strategy_mode TEXT DEFAULT 'BASELINE',
                    created_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filled_date TEXT,
                    filled_price REAL,
                    exit_date TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl REAL,
                    pnl_pct REAL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    portfolio_value REAL,
                    daily_pnl REAL,
                    daily_return REAL,
                    positions_opened INTEGER,
                    positions_closed INTEGER,
                    win_count INTEGER,
                    loss_count INTEGER
                )
            """)
            
            conn.commit()
    
    def create_position(self, position: Position) -> int:
        """Create a new position in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO positions 
                (ticker, order_type, entry_price, limit_price, stop_loss, take_profit,
                 shares, position_value, signal_score, strategy_mode, created_date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.ticker, position.order_type, position.entry_price,
                position.limit_price, position.stop_loss, position.take_profit,
                position.shares, position.position_value, position.signal_score,
                position.strategy_mode, position.created_date, position.status
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_open_positions(self) -> List[Position]:
        """Get all open/pending positions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM positions 
                WHERE status IN ('PENDING', 'OPEN', 'FLOATING_PROFIT', 'FLOATING_LOSS')
                ORDER BY created_date DESC
            """)
            
            rows = cursor.fetchall()
            return [Position(**dict(row)) for row in rows]
    
    def get_today_positions(self, today: Optional[str] = None) -> List[Position]:
        """Get positions created today."""
        today = today or date.today().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM positions 
                WHERE created_date = ?
                ORDER BY signal_score DESC
            """, (today,))
            
            rows = cursor.fetchall()
            return [Position(**dict(row)) for row in rows]
    
    def update_position_status(
        self,
        ticker: str,
        current_price: float,
        current_high: float,
        current_low: float
    ) -> Optional[Dict]:
        """
        Update position status based on current price.
        
        Returns exit signal if triggered.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get open position for ticker
            cursor.execute("""
                SELECT * FROM positions 
                WHERE ticker = ? AND status IN ('PENDING', 'OPEN', 'FLOATING_PROFIT', 'FLOATING_LOSS')
                ORDER BY created_date DESC LIMIT 1
            """, (ticker,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            pos = dict(row)
            result = {'changed': False, 'exit': False}
            
            # Check limit order fill
            if pos['status'] == 'PENDING' and pos['order_type'] == 'LIMIT':
                if current_low <= pos['limit_price'] <= current_high:
                    # Limit order filled
                    cursor.execute("""
                        UPDATE positions SET 
                            status = 'OPEN',
                            filled_date = ?,
                            filled_price = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (date.today().isoformat(), pos['limit_price'], pos['id']))
                    result['changed'] = True
                    result['filled'] = True
                    pos['status'] = 'OPEN'
                    pos['filled_price'] = pos['limit_price']
            
            # For open positions, check exit conditions
            if pos['status'] in ('OPEN', 'FLOATING_PROFIT', 'FLOATING_LOSS'):
                entry = pos['filled_price'] or pos['entry_price']
                
                # Check stop loss
                if current_low <= pos['stop_loss']:
                    pnl_pct = (pos['stop_loss'] - entry) / entry * 100
                    pnl = pnl_pct / 100 * pos['position_value']
                    
                    cursor.execute("""
                        UPDATE positions SET 
                            status = 'HIT_STOP_LOSS',
                            exit_date = ?,
                            exit_price = ?,
                            exit_reason = 'STOP_LOSS',
                            pnl = ?,
                            pnl_pct = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (date.today().isoformat(), pos['stop_loss'], pnl, pnl_pct, pos['id']))
                    
                    result['exit'] = True
                    result['reason'] = 'HIT_STOP_LOSS'
                    result['pnl_pct'] = pnl_pct
                
                # Check take profit
                elif current_high >= pos['take_profit']:
                    pnl_pct = (pos['take_profit'] - entry) / entry * 100
                    pnl = pnl_pct / 100 * pos['position_value']
                    
                    cursor.execute("""
                        UPDATE positions SET 
                            status = 'HIT_TAKE_PROFIT',
                            exit_date = ?,
                            exit_price = ?,
                            exit_reason = 'TAKE_PROFIT',
                            pnl = ?,
                            pnl_pct = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (date.today().isoformat(), pos['take_profit'], pnl, pnl_pct, pos['id']))
                    
                    result['exit'] = True
                    result['reason'] = 'HIT_TAKE_PROFIT'
                    result['pnl_pct'] = pnl_pct
                
                # Update floating P&L
                else:
                    pnl_pct = (current_price - entry) / entry * 100
                    new_status = 'FLOATING_PROFIT' if pnl_pct > 0 else 'FLOATING_LOSS'
                    
                    cursor.execute("""
                        UPDATE positions SET 
                            status = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (new_status, pos['id']))
                    
                    result['floating_pnl_pct'] = pnl_pct
            
            conn.commit()
            return result
    
    def expire_unfilled_orders(self, created_date: str) -> int:
        """Mark unfilled limit orders as ENTRY_NOT_HIT."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE positions SET 
                    status = 'ENTRY_NOT_HIT',
                    exit_reason = 'LIMIT_NOT_FILLED',
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = 'PENDING' AND created_date = ?
            """, (created_date,))
            
            conn.commit()
            return cursor.rowcount
    
    def close_position(
        self,
        ticker: str,
        exit_price: float,
        exit_reason: str
    ) -> Optional[Dict]:
        """Manually close a position."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM positions 
                WHERE ticker = ? AND status IN ('OPEN', 'FLOATING_PROFIT', 'FLOATING_LOSS')
                ORDER BY created_date DESC LIMIT 1
            """, (ticker,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            pos = dict(row)
            entry = pos['filled_price'] or pos['entry_price']
            pnl_pct = (exit_price - entry) / entry * 100
            pnl = pnl_pct / 100 * pos['position_value']
            
            cursor.execute("""
                UPDATE positions SET 
                    status = 'CLOSED',
                    exit_date = ?,
                    exit_price = ?,
                    exit_reason = ?,
                    pnl = ?,
                    pnl_pct = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (date.today().isoformat(), exit_price, exit_reason, pnl, pnl_pct, pos['id']))
            
            conn.commit()
            
            return {'ticker': ticker, 'pnl': pnl, 'pnl_pct': pnl_pct}
    
    def get_performance_summary(self) -> Dict:
        """Get overall trading performance."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get closed trades
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                       SUM(pnl) as total_pnl,
                       AVG(CASE WHEN pnl > 0 THEN pnl_pct END) as avg_win,
                       AVG(CASE WHEN pnl <= 0 THEN pnl_pct END) as avg_loss
                FROM positions
                WHERE status IN ('CLOSED', 'HIT_TAKE_PROFIT', 'HIT_STOP_LOSS')
            """)
            
            row = cursor.fetchone()
            
            return {
                'total_trades': row[0] or 0,
                'wins': row[1] or 0,
                'losses': row[2] or 0,
                'win_rate': (row[1] / row[0] * 100) if row[0] else 0,
                'total_pnl': row[3] or 0,
                'avg_win_pct': row[4] or 0,
                'avg_loss_pct': row[5] or 0
            }
    
    def clear_all_positions(self):
        """Clear all positions (for testing)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM positions")
            conn.commit()
