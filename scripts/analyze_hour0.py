#!/usr/bin/env python3
"""
Hour-0 Analysis Script
Analyzes hourly data to develop proxy features for daily data.
Run weekly to calibrate Hour-0 features.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import config
from data.storage import DataStorage
from data.intraday_fetcher import IntradayFetcher

console = Console()


def analyze_hour0_patterns(top_n: int = None, days: int = 60):
    """
    Analyze Hour-0 patterns from hourly data and store metrics.

    Args:
        top_n: Number of most liquid stocks to analyze (default: None = all tickers in universe)
        days: Days of hourly history to fetch (max 60)
    """
    storage = DataStorage(config.data.db_path)

    # Get stocks from universe
    console.print("\n[yellow]Step 1: Identifying stocks from universe...[/yellow]")
    import sqlite3
    with sqlite3.connect(storage.db_path) as conn:
        if top_n is None:
            # Get all tickers from universe
            query = """
                SELECT ticker, AVG(volume) as avg_volume
                FROM prices
                WHERE date >= date('now', '-60 days')
                GROUP BY ticker
                HAVING COUNT(*) > 40
                ORDER BY avg_volume DESC
            """
            liquid_stocks = conn.execute(query).fetchall()
        else:
            # Get top N liquid stocks
            query = """
                SELECT ticker, AVG(volume) as avg_volume
                FROM prices
                WHERE date >= date('now', '-60 days')
                GROUP BY ticker
                HAVING COUNT(*) > 40
                ORDER BY avg_volume DESC
                LIMIT ?
            """
            liquid_stocks = conn.execute(query, (top_n,)).fetchall()

        tickers = [row[0] for row in liquid_stocks]

    console.print(Panel.fit(
        "[bold cyan]Hour-0 Pattern Analysis[/bold cyan]\n"
        f"[dim]Analyzing {len(tickers)} stocks over {days} days[/dim]",
        border_style="cyan"
    ))

    console.print(f"  [green]✓ Selected {len(tickers)} liquid stocks[/green]")

    # Fetch hourly data with rate limiting
    console.print(f"\n[yellow]Step 2: Fetching hourly data (rate-limited)...[/yellow]")
    fetcher = IntradayFetcher(batch_size=10, delay_seconds=3)
    hourly_data = fetcher.fetch_hourly_batch(tickers, days=days)

    if hourly_data.empty:
        console.print("[red]✗ Failed to fetch hourly data[/red]")
        return

    console.print(f"  [green]✓ Fetched {len(hourly_data)} hourly bars[/green]")

    # Calculate Hour-0 metrics
    console.print("\n[yellow]Step 3: Calculating Hour-0 metrics...[/yellow]")
    h0_metrics = fetcher.calculate_hour0_metrics(hourly_data)

    if h0_metrics.empty:
        console.print("[red]✗ Failed to calculate Hour-0 metrics[/red]")
        return

    console.print(f"  [green]✓ Calculated metrics for {len(h0_metrics)} trading days[/green]")

    # Store in database
    console.print("\n[yellow]Step 4: Storing Hour-0 metrics...[/yellow]")
    with sqlite3.connect(storage.db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS hour0_metrics")
        conn.execute("""
            CREATE TABLE hour0_metrics (
                ticker TEXT,
                date DATE,
                h0_spike_pct REAL,
                h0_fade_pct REAL,
                h0_net_pct REAL,
                h0_spike_is_day_high INTEGER,
                h0_spike_to_close REAL,
                PRIMARY KEY (ticker, date)
            )
        """)

        h0_metrics.to_sql('hour0_metrics', conn, if_exists='append', index=False)
        conn.commit()

    console.print(f"  [green]✓ Stored {len(h0_metrics)} Hour-0 metrics[/green]")

    # Generate analysis summary
    console.print("\n[yellow]Step 5: Generating analysis...[/yellow]")

    # Calculate statistics
    stats = {
        'avg_spike': h0_metrics['h0_spike_pct'].mean(),
        'avg_fade': h0_metrics['h0_fade_pct'].mean(),
        'spike_positive_pct': (h0_metrics['h0_spike_pct'] > 0).sum() / len(h0_metrics),
        'fade_negative_pct': (h0_metrics['h0_fade_pct'] < 0).sum() / len(h0_metrics),
        'spike_is_high_pct': h0_metrics['h0_spike_is_day_high'].mean(),
    }

    # Display results
    table = Table(title="Hour-0 Pattern Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Avg Spike (9-10 AM)", f"{stats['avg_spike']:.2%}")
    table.add_row("Avg Fade (10-11 AM)", f"{stats['avg_fade']:.2%}")
    table.add_row("Spike > 0% (Days)", f"{stats['spike_positive_pct']:.1%}")
    table.add_row("Fade < 0% (Days)", f"{stats['fade_negative_pct']:.1%}")
    table.add_row("10 AM = Day High", f"{stats['spike_is_high_pct']:.1%}")

    console.print("\n")
    console.print(table)

    console.print("\n[bold green]✓ Hour-0 analysis complete![/bold green]")
    console.print("[dim]These metrics are now available for feature engineering.[/dim]")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Hour-0 patterns from intraday data')
    parser.add_argument('--stocks', type=int, default=None, help='Number of stocks to analyze (default: None = all tickers in universe)')
    parser.add_argument('--days', type=int, default=60, help='Days of history (max 60)')

    args = parser.parse_args()

    analyze_hour0_patterns(top_n=args.stocks, days=args.days)


if __name__ == "__main__":
    main()
