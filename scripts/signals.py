#!/usr/bin/env python3
"""
Stock Prediction Signals (LSTM Edition)
Generates trading signals for today using pre-market data.
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from ml.model import ModelWrapper
from ml.features import FeatureEngineer
from utils.logger import TimedLogger
from data.blacklist import BLACKLIST_UNIVERSE

console = Console()

def load_ticker_history(ticker, limit=200):
    conn = sqlite3.connect(config.data.db_path)
    query = f"""
    SELECT date, open, high, low, close, volume
    FROM prices
    WHERE ticker = '{ticker}'
    ORDER BY date DESC
    LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.iloc[::-1] # Reverse to chronological order

def main():
    import argparse
    parser = argparse.ArgumentParser(description='IDX Stock Predictions')
    parser.add_argument('--capital', type=float, default=0.0, help='Total capital to allocate (IDR)')
    parser.add_argument('--num-stock', type=int, default=0, help='Number of stocks to buy')
    parser.add_argument('--fetch-latest', action='store_true', help='Fetch latest data from Yahoo Finance for today')
    args = parser.parse_args()

    logger = TimedLogger()

    # Fetch latest data if requested
    if args.fetch_latest:
        logger.log("[cyan]Fetching latest market data from Yahoo Finance...[/cyan]")
        console.print("\n[yellow]Fetching latest data from Yahoo Finance...[/yellow]")

        try:
            from data.fetcher import DataFetcher
            from data.storage import DataStorage
            from data.universe import IDX_UNIVERSE

            fetcher = DataFetcher(IDX_UNIVERSE)
            storage = DataStorage(config.data.db_path)

            # Fetch last 5 days to ensure we have today's data
            data = fetcher.fetch_batch(days=5)

            if not data.empty:
                count = storage.upsert_prices(data)
                logger.log(f"[green]✓ Updated {count} price records[/green]")
                console.print(f"[green]✓ Updated {count} price records from Yahoo Finance[/green]")
            else:
                logger.log("[yellow]No new data fetched[/yellow]")
                console.print("[yellow]No new data available[/yellow]")
        except Exception as e:
            logger.log(f"[red]Failed to fetch latest data: {e}[/red]")
            console.print(f"[red]Warning: Failed to fetch latest data: {e}[/red]")
            console.print("[yellow]Continuing with existing database data...[/yellow]")

    console.print(Panel.fit(
        "[bold cyan]IDX Stock Predictions (LSTM)[/bold cyan]\n"
        "[dim]Generating buy signals with ML confidence[/dim]",
        border_style="cyan"
    ))

    # Display configuration
    console.print("\n[bold]Configuration:[/bold]")
    if args.capital > 0:
        console.print(f"  Capital to Allocate: Rp {args.capital:,.0f}")
    else:
        console.print(f"  Capital: Not specified (allocation disabled)")

    if args.num_stock > 0:
        console.print(f"  Number of Stocks:    {args.num_stock}")
    else:
        console.print(f"  Number of Stocks:    Not specified (show all signals)")

    # 2. Load Model
    logger.log("[cyan]Loading LSTM model...[/cyan]")
    try:
        model = ModelWrapper(config)
        epoch, metrics = model.load_checkpoint("models/best_lstm.pt")
        if epoch is None:
             logger.log("[yellow]Warning: Could not load model checkpoint[/yellow]")
             console.print("[yellow]Warning: Could not load model checkpoint. Ensure 'models/best_lstm.pt' exists.[/yellow]")
             return
        logger.log(f"[green]✓ Model loaded:[/green] Epoch {epoch}, Val Acc: {metrics.get('val_acc', 0):.1%}")
        console.print(f"\n[dim]Loaded LSTM Model (Epoch {epoch}, Acc: {metrics.get('val_acc', 0):.1%})[/dim]")
    except Exception as e:
        logger.log(f"[red]Failed to load model: {e}[/red]")
        console.print(f"[red]Failed to load model: {e}[/red]")
        return

    feature_eng = FeatureEngineer(config)

    # Get active universe
    from data.universe import IDX_UNIVERSE
    tickers = IDX_UNIVERSE

    signals = []

    import warnings
    warnings.filterwarnings('ignore')

    logger.log(f"[cyan]Scanning {len(tickers)} tickers for signals...[/cyan]")
    with console.status(f"[yellow]Scanning {len(tickers)} tickers...[/yellow]"):
        for ticker in tickers:
            # Filter blacklist
            if ticker in BLACKLIST_UNIVERSE:
                continue

            try:
                df = load_ticker_history(ticker, limit=config.data.window_size + 10)
                if len(df) < config.data.window_size:
                    continue

                # Prepare inference sequence
                X_tensor = feature_eng.prepare_inference(df)

                # Predict
                pred_class, probs = model.predict(X_tensor)
                pred_class = pred_class[0]
                probs = probs[0]

                # Class 2 is Profit
                buy_prob = probs[2]

                # Thresholds: Class 2 AND Prob > 50%
                if pred_class == 2 and buy_prob > 0.5:
                    latest = df.iloc[-1]
                    price = latest['close']

                    signals.append({
                        'Ticker': ticker,
                        'Price': price,
                        'Conf': buy_prob,
                        'SL': price * (1 - config.ml.tbl_barrier),
                        'TP': price * (1 + config.ml.tbl_barrier)
                    })

            except Exception:
                continue

    logger.log(f"[green]✓ Scanning complete:[/green] Found {len(signals)} buy signals")

    if not signals:
        console.print("\n[yellow]No signals found matching criteria (Class=PROFIT, Confidence>50%)[/yellow]")
        return

    # Sort by confidence (highest first)
    signals.sort(key=lambda x: x['Conf'], reverse=True)

    # Determine which signals to show
    has_allocation = args.capital > 0 and args.num_stock > 0

    if has_allocation:
        # Take top N stocks for allocation
        top_N = min(args.num_stock, len(signals))
        signals_to_allocate = signals[:top_N]

        # Calculate confidence-weighted allocation
        total_confidence = sum(s['Conf'] for s in signals_to_allocate)

        for signal in signals_to_allocate:
            # Allocate proportionally to confidence
            weight = signal['Conf'] / total_confidence
            allocation = args.capital * weight

            # Calculate shares (round to lots of 100)
            shares = int(allocation / signal['Price'] / 100) * 100
            actual_allocation = shares * signal['Price']

            # Estimated P/L based on TBL barriers
            estimated_profit = actual_allocation * config.ml.tbl_barrier  # +3%
            estimated_loss = actual_allocation * config.ml.tbl_barrier    # -3%

            signal['Allocation'] = actual_allocation
            signal['Shares'] = shares
            signal['Est_Profit'] = estimated_profit
            signal['Est_Loss'] = estimated_loss

        # Show top N allocated + a few more for watchlist
        display_signals = signals[:min(top_N + 5, len(signals))]
    else:
        # Show all signals (or top 20)
        display_signals = signals[:20]

    # Display Results Table
    table = Table(title=f"Buy Signals for {datetime.now().strftime('%Y-%m-%d')}")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Ticker", style="cyan")
    table.add_column("Price", justify="right")
    table.add_column("Conf", justify="right", style="magenta")
    table.add_column("SL/TP", justify="center", style="dim")

    if has_allocation:
        table.add_column("Shares", justify="right")
        table.add_column("Allocation", justify="right", style="yellow")
        table.add_column("Est P/L", justify="right", style="green")

    for i, signal in enumerate(display_signals, 1):
        is_allocated = has_allocation and i <= args.num_stock
        rank_style = "bold cyan" if is_allocated else "dim"

        # Price formatting
        price_str = f"Rp {signal['Price']:,.0f}"

        # SL/TP
        sl_tp_str = f"{signal['SL']:.0f} / {signal['TP']:.0f}"

        # Confidence
        conf_str = f"{signal['Conf']:.1%}"

        if has_allocation and is_allocated:
            # Show allocation details
            shares_str = f"{signal['Shares']:,}"
            alloc_str = f"Rp {signal['Allocation']:,.0f}"
            pl_str = f"+{signal['Est_Profit']/1e6:.2f}M / -{signal['Est_Loss']/1e6:.2f}M"

            table.add_row(
                f"[{rank_style}]{i}[/{rank_style}]",
                f"[{rank_style}]{signal['Ticker']}[/{rank_style}]",
                price_str,
                conf_str,
                sl_tp_str,
                shares_str,
                alloc_str,
                pl_str
            )
        else:
            # Watchlist or no allocation mode
            if has_allocation:
                table.add_row(
                    f"[{rank_style}]{i}[/{rank_style}]",
                    f"[{rank_style}]{signal['Ticker']}[/{rank_style}]",
                    price_str,
                    conf_str,
                    sl_tp_str,
                    "-",
                    "-",
                    "-"
                )
            else:
                table.add_row(
                    f"{i}",
                    signal['Ticker'],
                    price_str,
                    conf_str,
                    sl_tp_str
                )

    console.print("\n")
    console.print(table)

    # Summary
    if has_allocation:
        total_allocated = sum(s.get('Allocation', 0) for s in signals_to_allocate)
        total_est_profit = sum(s.get('Est_Profit', 0) for s in signals_to_allocate)
        total_est_loss = sum(s.get('Est_Loss', 0) for s in signals_to_allocate)

        console.print(f"\n[bold]Allocation Summary:[/bold]")
        console.print(f"  Total Capital:      Rp {args.capital:,.0f}")
        console.print(f"  Actually Allocated: Rp {total_allocated:,.0f} ({total_allocated/args.capital*100:.1f}%)")
        console.print(f"  Stocks to Buy:      {len(signals_to_allocate)}")
        console.print(f"  Est. Profit (3%):   [green]+Rp {total_est_profit:,.0f} (+{total_est_profit/1e6:.2f}M)[/green]")
        console.print(f"  Est. Loss (3%):     [red]-Rp {total_est_loss:,.0f} (-{total_est_loss/1e6:.2f}M)[/red]")

    logger.log(f"[green]✓ Analysis complete[/green]")

if __name__ == "__main__":
    main()
