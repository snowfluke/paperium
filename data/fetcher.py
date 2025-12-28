"""
Data Fetcher Module
Fetches OHLCV data for Indonesian stocks using Yahoo Finance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
import os
import pickle
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches stock data from Yahoo Finance for Indonesian stocks.
    Indonesian stocks use .JK suffix (e.g., BBCA.JK for Bank Central Asia)
    """
    
    def __init__(self, tickers: List[str], cache_dir: str = ".cache"):
        """
        Initialize fetcher with list of tickers.
        
        Args:
            tickers: List of stock tickers with .JK suffix
            cache_dir: Directory for caching fetched data
        """
        self.tickers = sorted(tickers)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    
    def fetch_historical(
        self, 
        days: int = 730,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV data for all tickers.
        
        Args:
            days: Number of days of history to fetch
            end_date: End date (default: today)
            
        Returns:
            Dictionary mapping ticker to DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching data for {len(self.tickers)} tickers from {start_date.date()} to {end_date.date()}")
        
        data = {}
        failed = []
        
        for ticker in self.tickers:
            try:
                df = self._fetch_single(ticker, start_date, end_date)
                if df is not None and len(df) > 0:
                    data[ticker] = df
                    logger.info(f"✓ {ticker}: {len(df)} rows")
                else:
                    failed.append(ticker)
            except Exception as e:
                logger.warning(f"✗ {ticker}: {str(e)}")
                failed.append(ticker)
        
        if failed:
            logger.warning(f"Failed to fetch: {failed}")
        
        logger.info(f"Successfully fetched {len(data)}/{len(self.tickers)} tickers")
        return data
    
    def _fetch_single(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker.
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty:
            return None
        
        # Standardize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Add ticker column
        df['ticker'] = ticker
        
        # Reset index to make date a column
        df = df.reset_index()
        df = df.rename(columns={'Date': 'date'})
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        return df
    
    def fetch_latest(self, days: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Fetch latest data (for daily updates).
        
        Args:
            days: Number of recent days to fetch
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        return self.fetch_historical(days=days)
    
    def fetch_batch(
        self, 
        days: int = 730,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch all tickers and combine into single DataFrame.
        More efficient for bulk downloads.
        
        Args:
            days: Number of days of history
            end_date: End date
            
        Returns:
            Combined DataFrame with all tickers
        """
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=days)
        
        # 1. Caching Logic
        # Cache key includes dates and hash of tickers to handle universe changes
        ticker_hash = hashlib.md5("".join(self.tickers).encode()).hexdigest()[:8]
        # Hourly granularity: captures pre-market vs post-market difference
        cache_key = f"fetch_{start_date.date()}_{end_date.date()}_{end_date.strftime('%Y%m%d_%H')}_{ticker_hash}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_df = pickle.load(f)
                logger.info(f"✓ Using cached data from {cache_path} ({len(cached_df)} rows)")
                return cached_df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        logger.info(f"Fetching latest market data for {end_date.strftime('%Y-%m-%d %H:%M:%S')}...")
        logger.info(f"Batch fetching {len(self.tickers)} tickers...")
        
        try:
            # Download all at once (more efficient)
            # Download all at once
            # Note: threads=True can cause NoneType errors in some environments/versions
            df = yf.download(
                self.tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                group_by='ticker',
                threads=False, # Disable threading to fix the NoneType subscriptable error
                progress=True
            )

            if df is None or df.empty:
                logger.error("Batch download returned empty or None DataFrame")
                return pd.DataFrame()
            
            # Restructure the multi-level column DataFrame
            records = []
            
            for ticker in self.tickers:
                try:
                    if ticker in df.columns.get_level_values(0):
                        ticker_df = df[ticker].copy()
                        ticker_df = ticker_df.dropna()
                        
                        if len(ticker_df) > 0:
                            ticker_df = ticker_df.reset_index()
                            # Use current column names to avoid mapping errors if structure changes
                            cols = [c.lower() for c in ticker_df.columns]
                            ticker_df.columns = cols
                            
                            # Standardize names (order might vary)
                            rename_map = {'price': 'close', 'adj close': 'close'}
                            ticker_df = ticker_df.rename(columns=rename_map)
                            
                            ticker_df['ticker'] = ticker
                            ticker_df['date'] = pd.to_datetime(ticker_df['date']).dt.tz_localize(None)
                            records.append(ticker_df)
                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {e}")
            
            if records:
                combined = pd.concat(records, ignore_index=True)
                # Reorder columns
                combined = combined[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
                
                # Save to cache
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(combined, f)
                except:
                    pass
                    
                logger.info(f"Batch fetch complete: {len(combined)} total rows")
                return combined
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            # Fallback to individual fetching
            data = self.fetch_historical(days=days, end_date=end_date)
            if data:
                return pd.concat(data.values(), ignore_index=True)
            return pd.DataFrame()


def get_sector_mapping() -> Dict[str, str]:
    """
    Returns sector mapping for Indonesian stocks.
    Useful for sector diversification.
    """
    return {
        # Banking
        "BBCA.JK": "Banking", "BBRI.JK": "Banking", "BMRI.JK": "Banking",
        "BBNI.JK": "Banking", "BRIS.JK": "Banking",
        # Telco
        "TLKM.JK": "Telecom", "EXCL.JK": "Telecom", "ISAT.JK": "Telecom",
        # Consumer
        "UNVR.JK": "Consumer", "ICBP.JK": "Consumer", "INDF.JK": "Consumer",
        "MYOR.JK": "Consumer", "KLBF.JK": "Healthcare",
        # Mining & Energy
        "ADRO.JK": "Mining", "PTBA.JK": "Mining", "ITMG.JK": "Mining",
        "MEDC.JK": "Energy", "PGAS.JK": "Energy",
        "MDKA.JK": "Mining", "ANTM.JK": "Mining", "INCO.JK": "Mining", "TINS.JK": "Mining",
        # Industrials
        "ASII.JK": "Automotive", "UNTR.JK": "Machinery", "SRIL.JK": "Textile",
        # Property & Construction
        "SMGR.JK": "Materials", "WIKA.JK": "Construction", "PTPP.JK": "Construction",
        "BSDE.JK": "Property",
        # Others
        "GGRM.JK": "Tobacco", "HMSP.JK": "Tobacco",
        "ERAA.JK": "Retail", "ACES.JK": "Retail", "MAPI.JK": "Retail",
        "AKRA.JK": "Energy", "TOWR.JK": "Infrastructure", "TBIG.JK": "Infrastructure",
        "JSMR.JK": "Infrastructure",
        # Technology
        "GOTO.JK": "Technology", "BUKA.JK": "Technology",
        "EMTK.JK": "Media", "SCMA.JK": "Media"
    }
