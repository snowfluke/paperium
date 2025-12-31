#!/usr/bin/env python3
"""
Download IHSG Index Data
Fetches ^JKSE (Indonesia Composite Index) data for market regime detection
"""
import sys
import os
import logging
from rich.console import Console

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetcher import DataFetcher
from data.storage import DataStorage
from config import config
from utils.logger import TimedLogger

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_ihsg_data(days: int = 1825):  # 5 years default
    """
    Download IHSG (^JKSE) index data.

    Args:
        days: Number of days of historical data to fetch
    """
    timed_logger = TimedLogger()
    timed_logger.log("Starting IHSG data download")
    console.print("[bold cyan]Downloading IHSG Index Data (^JKSE)[/bold cyan]")

    # Create fetcher for IHSG
    fetcher = DataFetcher(["^JKSE"])
    storage = DataStorage(config.data.db_path)

    console.print(f"Fetching {days} days of IHSG history...")
    timed_logger.log(f"Fetching {days} days of IHSG data")

    # Fetch data
    data = fetcher.fetch_batch(days=days)

    if not data.empty:
        timed_logger.log(f"Data fetched: {len(data)} records")
        console.print(f"Downloaded {len(data)} records for IHSG")
        console.print(f"Date range: {data['date'].min()} to {data['date'].max()}")

        # Store in database
        timed_logger.log("Saving to database")
        count = storage.upsert_prices(data)
        timed_logger.log("IHSG download complete")
        console.print(f"[bold green]✓ IHSG data saved. {count} records updated.[/bold green]")

        # Show summary statistics
        console.print("\n[bold]IHSG Summary:[/bold]")
        console.print(f"  Close range: {data['close'].min():.2f} - {data['close'].max():.2f}")
        console.print(f"  Average volume: {data['volume'].mean():,.0f}")
    else:
        console.print("[bold red]✕ Failed to download IHSG data[/bold red]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download IHSG index data")
    parser.add_argument("--days", type=int, default=1825, help="Days of history to fetch (default: 1825 = 5 years)")
    args = parser.parse_args()

    download_ihsg_data(days=args.days)
