#!/usr/bin/env python3
import sys
import os
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from data.fetcher import DataFetcher
from data.storage import DataStorage
from utils.logger import TimedLogger

console = Console()
logging.basicConfig(level=logging.INFO)

def sync_data():
    logger = TimedLogger()
    logger.log("Starting universe data sync")
    console.print("[bold cyan]Starting Universe Data Sync[/bold cyan]")
    storage = DataStorage(config.data.db_path)
    fetcher = DataFetcher(config.data.stock_universe)
    
    total_tickers = len(config.data.stock_universe)
    lookback = config.data.lookback_days
    console.print(f"Syncing [bold]{total_tickers}[/bold] tickers ({lookback} days history)...")
    
    # Use fetch_batch but track progress
    # fetch_batch uses yfinance's internal progress bar, but we'll wrap it to be sure
    logger.log(f"Fetching data for {total_tickers} tickers ({lookback} days)")
    data = fetcher.fetch_batch(days=lookback)
    
    if not data.empty:
        logger.log(f"Data fetched: {len(data):,} rows")
        console.print(f"Upserting {len(data)} records to database...")
        logger.log("Upserting to database")
        count = storage.upsert_prices(data)
        logger.log(f"Sync complete: {count} records updated")
        console.print(f"[bold green]✓ Database Sync Complete. {count} records updated.[/bold green]")
    else:
        console.print("[bold red]✕ Data fetch failed or returned empty.[/bold red]")


if __name__ == "__main__":
    sync_data()
