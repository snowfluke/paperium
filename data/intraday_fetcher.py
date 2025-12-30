"""
Intraday Data Fetcher Module
Safely fetches hourly data for Hour-0 feature engineering with rate limiting.
"""
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

logger = logging.getLogger(__name__)
console = Console()


class IntradayFetcher:
    """
    Safely fetches hourly intraday data with rate limiting.
    Only fetches last 60 days (Yahoo Finance limit for hourly data).
    """

    def __init__(self, batch_size: int = 10, delay_seconds: int = 3):
        """
        Initialize intraday fetcher with rate limiting.

        Args:
            batch_size: Number of stocks to fetch per batch (default 10)
            delay_seconds: Seconds to wait between batches (default 3)
        """
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds

    def fetch_hourly_batch(
        self,
        tickers: List[str],
        days: int = 60
    ) -> pd.DataFrame:
        """
        Safely fetch hourly data for multiple tickers with rate limiting.

        Args:
            tickers: List of stock tickers
            days: Days of history (max 60 for hourly)

        Returns:
            DataFrame with hourly OHLCV data
        """
        if days > 60:
            console.print("[yellow]⚠ Yahoo Finance only provides ~60 days of hourly data. Using 60 days.[/yellow]")
            days = 60

        all_data = []
        total_batches = (len(tickers) + self.batch_size - 1) // self.batch_size

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Fetching hourly data ({self.batch_size} stocks/batch, {self.delay_seconds}s delay)...",
                total=len(tickers)
            )

            for i in range(0, len(tickers), self.batch_size):
                batch = tickers[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1

                try:
                    # Fetch this batch
                    for ticker in batch:
                        try:
                            stock = yf.Ticker(ticker)
                            df = stock.history(period=f'{days}d', interval='1h')

                            if not df.empty:
                                df['ticker'] = ticker
                                df['date'] = df.index
                                all_data.append(df)

                        except Exception as e:
                            logger.debug(f"Failed to fetch hourly data for {ticker}: {e}")

                        progress.update(task, advance=1)

                    # Rate limiting: Wait between batches (except last batch)
                    if batch_num < total_batches:
                        time.sleep(self.delay_seconds)

                except Exception as e:
                    logger.warning(f"Batch {batch_num} failed: {e}")
                    progress.update(task, advance=len(batch))

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)

        # Standardize columns
        result.columns = [col.lower() for col in result.columns]
        result = result.rename(columns={
            'datetime': 'timestamp'
        })

        return result

    def calculate_hour0_metrics(self, hourly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Hour-0 (9-11 AM) metrics from hourly data.

        Args:
            hourly_df: DataFrame with hourly OHLCV data

        Returns:
            DataFrame with daily Hour-0 metrics per ticker
        """
        if hourly_df.empty:
            return pd.DataFrame()

        results = []

        for ticker in hourly_df['ticker'].unique():
            ticker_data = hourly_df[hourly_df['ticker'] == ticker].copy()
            ticker_data['hour'] = pd.to_datetime(ticker_data['date']).dt.hour
            ticker_data['trade_date'] = pd.to_datetime(ticker_data['date']).dt.date

            # Group by trading day
            for trade_date, day_data in ticker_data.groupby('trade_date'):
                try:
                    # Get 9 AM, 10 AM, 11 AM bars
                    h9 = day_data[day_data['hour'] == 9]
                    h10 = day_data[day_data['hour'] == 10]
                    h11 = day_data[day_data['hour'] == 11]

                    if h9.empty or h10.empty or h11.empty:
                        continue

                    h9_open = h9['open'].iloc[0]
                    h10_high = h10['high'].iloc[0]
                    h10_close = h10['close'].iloc[0]
                    h11_close = h11['close'].iloc[0]

                    # Get daily metrics
                    daily_high = day_data['high'].max()
                    daily_close = day_data['close'].iloc[-1]

                    # Calculate Hour-0 metrics
                    metrics = {
                        'ticker': ticker,
                        'date': trade_date,
                        'h0_spike_pct': (h10_high - h9_open) / h9_open if h9_open > 0 else 0,
                        'h0_fade_pct': (h11_close - h10_high) / h10_high if h10_high > 0 else 0,
                        'h0_net_pct': (h10_close - h9_open) / h9_open if h9_open > 0 else 0,
                        'h0_spike_is_day_high': 1 if abs(h10_high - daily_high) / daily_high < 0.01 else 0,
                        'h0_spike_to_close': (h10_high - daily_close) / h10_high if h10_high > 0 else 0,
                    }

                    results.append(metrics)

                except Exception as e:
                    logger.debug(f"Failed to calculate Hour-0 for {ticker} on {trade_date}: {e}")

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)
