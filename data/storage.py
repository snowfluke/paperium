"""
Data Storage Module
SQLite-based persistence for historical data and model state
"""
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Optional
import logging
import os

logger = logging.getLogger(__name__)


class DataStorage:
    """
    SQLite-based storage for OHLCV data and trading signals.
    """
    
    def __init__(self, db_path: str = "data/ihsg_trading.db"):
        """
        Initialize storage with database path.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Price data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, ticker)
                )
            """)
            
            # Signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    signal_type TEXT,
                    score REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    position_size REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, ticker, signal_type)
                )
            """)
            
            # Trades table (for tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_date TEXT,
                    exit_price REAL,
                    position_size REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    exit_reason TEXT,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    model_version TEXT,
                    accuracy REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_return REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date)")
            
            conn.commit()
    
    def save_prices(self, df: pd.DataFrame) -> int:
        """
        Save price data to database.
        
        Args:
            df: DataFrame with columns [date, ticker, open, high, low, close, volume]
            
        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            return 0
        
        # Ensure date is string format
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            # Use INSERT OR REPLACE for upsert behavior
            rows_before = pd.read_sql("SELECT COUNT(*) as cnt FROM prices", conn)['cnt'].iloc[0]
            
            df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']].to_sql(
                'prices',
                conn,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            rows_after = pd.read_sql("SELECT COUNT(*) as cnt FROM prices", conn)['cnt'].iloc[0]
            inserted = rows_after - rows_before
            
        logger.info(f"Saved {inserted} price records to database")
        return inserted
    
    def upsert_prices(self, df: pd.DataFrame) -> int:
        """
        Upsert (insert or update) price data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Number of rows affected
        """
        if df.empty:
            return 0
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            count = 0
            
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO prices 
                    (date, ticker, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['date'], row['ticker'], 
                    row['open'], row['high'], row['low'], row['close'], 
                    int(row['volume']) if pd.notna(row['volume']) else 0
                ))
                count += 1
            
            conn.commit()
        
        logger.info(f"Upserted {count} price records")
        return count
    
    def get_prices(
        self, 
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve price data from database.
        
        Args:
            tickers: List of tickers to filter (None for all)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with price data
        """
        query = "SELECT date, ticker, open, high, low, close, volume FROM prices WHERE 1=1"
        params = []
        
        if tickers:
            placeholders = ','.join(['?' for _ in tickers])
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date, ticker"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, params=params)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Get all data for a single ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            DataFrame with price data for ticker
        """
        return self.get_prices(tickers=[ticker])
    
    def get_latest_date(self, ticker: Optional[str] = None) -> Optional[str]:
        """
        Get the most recent date in the database.
        
        Args:
            ticker: Optional ticker to check
            
        Returns:
            Latest date as string or None
        """
        query = "SELECT MAX(date) as max_date FROM prices"
        params = []
        
        if ticker:
            query += " WHERE ticker = ?"
            params.append(ticker)
        
        with sqlite3.connect(self.db_path) as conn:
            result = pd.read_sql(query, conn, params=params)
        
        if result.empty or result['max_date'].iloc[0] is None:
            return None
        
        return result['max_date'].iloc[0]
    
    def is_data_fresh(self, target_date: Optional[str] = None, min_tickers: int = 50) -> bool:
        """
        Check if data for the target date is already in the database.
        
        Args:
            target_date: Date to check (YYYY-MM-DD). Defaults to today.
            min_tickers: Minimum number of tickers required to consider data "fresh"
            
        Returns:
            True if data is fresh and sufficient, False otherwise
        """
        from datetime import date as dt_date
        
        if target_date is None:
            target_date = dt_date.today().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            result = pd.read_sql(
                "SELECT COUNT(DISTINCT ticker) as cnt FROM prices WHERE date = ?",
                conn,
                params=[target_date]
            )
        
        ticker_count = result['cnt'].iloc[0] if not result.empty else 0
        return ticker_count >= min_tickers
    
    def save_signals(self, df: pd.DataFrame) -> int:
        """
        Save trading signals to database.
        
        Args:
            df: DataFrame with signal data
            
        Returns:
            Number of rows saved
        """
        if df.empty:
            return 0
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            count = 0
            
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO signals 
                    (date, ticker, signal_type, score, entry_price, stop_loss, take_profit, position_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['date'], row['ticker'], row.get('signal_type', 'composite'),
                    row.get('score'), row.get('entry_price'), row.get('stop_loss'),
                    row.get('take_profit'), row.get('position_size')
                ))
                count += 1
            
            conn.commit()
        
        return count
    
    def get_signals(self, date: str) -> pd.DataFrame:
        """
        Get signals for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with signals
        """
        query = """
            SELECT date, ticker, signal_type, score, entry_price, stop_loss, take_profit, position_size
            FROM signals
            WHERE date = ?
            ORDER BY score DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, params=[date])
        
        return df
    
    def get_ticker_count(self) -> int:
        """Get number of unique tickers in database."""
        with sqlite3.connect(self.db_path) as conn:
            result = pd.read_sql("SELECT COUNT(DISTINCT ticker) as cnt FROM prices", conn)
        return result['cnt'].iloc[0]
    
    def get_date_range(self) -> tuple:
        """Get date range of data in database."""
        with sqlite3.connect(self.db_path) as conn:
            result = pd.read_sql(
                "SELECT MIN(date) as min_date, MAX(date) as max_date FROM prices", 
                conn
            )
        return (result['min_date'].iloc[0], result['max_date'].iloc[0])
