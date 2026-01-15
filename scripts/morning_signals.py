#!/usr/bin/env python3
"""
Morning Signals Script
Run before market open to get today's trading signals

Usage:
    uv run python scripts/morning_signals.py

Or if installed:
    uv run morning
"""

import sys
import os
import json
import pickle
import hashlib
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import config
from data.fetcher import DataFetcher, get_sector_mapping
from data.storage import DataStorage
from signals.combiner import SignalCombiner
from signals.screener import Screener
from ml.model import TradingModel
from strategy.exit_manager import ExitManager
from strategy.position_sizer import PositionSizer
from strategy.position_manager import PositionManager, Position, OrderType, PositionStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()


# Thresholds for order type decision
MARKET_ORDER_THRESHOLD = 0.70  # Lowered slightly
LIMIT_ORDER_THRESHOLD = 0.30   # Minimum filter threshold
MIN_SIGNAL_SCORE = 0.30        # Increased to filter out weak/noisy signals (e.g. 20%)


class MorningSignals:
    """Generates morning trading signals and manages existing positions."""

    def __init__(self):
        self.storage = DataStorage(config.data.db_path)
        self.fetcher = DataFetcher(config.data.stock_universe)
        self.signal_combiner = SignalCombiner(config)
        self.screener = Screener(config)

        # Load Trading Model (XGBoost)
        self.model = TradingModel(config.ml)
        xgb_path = os.path.join("models", "global_xgb_champion.pkl")
        if os.path.exists(xgb_path):
            self.model.load(xgb_path)
        else:
            logger.warning("No champion model found! Using indicators only.")
        self.exit_manager = ExitManager(config.exit)
        self.position_sizer = PositionSizer(config.portfolio)
        self.position_manager = PositionManager()
        self.sector_mapping = get_sector_mapping()
        self.metadata = self._load_metadata()

    def _cleanup_old_caches(self, cache_dir: Path, current_hour: str):
        """Remove cache files older than current hour."""
        try:
            # Clean up signals cache files (format: signals_YYYYMMDD_HH_hash.pkl)
            for cache_file in cache_dir.glob("signals_*.pkl"):
                parts = cache_file.stem.split('_')
                if len(parts) >= 3:
                    file_hour = f"{parts[1]}_{parts[2]}"  # YYYYMMDD_HH
                    if file_hour < current_hour:
                        cache_file.unlink()
                        logger.debug(f"Cleaned up old cache: {cache_file.name}")

            # Clean up fetch cache files (format: fetch_DATE_DATE_YYYYMMDD_HH_hash.pkl)
            for cache_file in cache_dir.glob("fetch_*.pkl"):
                parts = cache_file.stem.split('_')
                if len(parts) >= 5:
                    # Extract YYYYMMDD_HH (parts[3] and parts[4])
                    file_hour = f"{parts[3]}_{parts[4]}"  # YYYYMMDD_HH
                    if file_hour < current_hour:
                        cache_file.unlink()
                        logger.debug(f"Cleaned up old cache: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")

    def _load_metadata(self) -> Dict:
        """Load model metadata from JSON."""
        metadata_path = 'models/champion_metadata.json'
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f).get('xgboost', {})
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}

    def run(self, portfolio_value: Optional[float] = None, custom_capital: float = 0.0):
        """
        Run morning signal generation.

        Args:
            portfolio_value: Total portfolio value (optional, falls back to config)
            custom_capital: Extra capital to distribute among signals
        """
        if portfolio_value is None:
            portfolio_value = config.portfolio.total_value

        self.custom_capital = custom_capital

        # Get dynamic metrics
        win_rate = self.metadata.get('win_rate', 0.0)
        trained_date = self.metadata.get('date', 'Unknown')

        console.print()
        console.print(Panel(
            f"[bold white]IHSG MORNING SIGNALS[/bold white]\n"
            f"[dim]Quantitative Trading Dashboard • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n"
            f"[dim]Model Win Rate: [bold green]{win_rate:.1%}[/bold green] | Trained: [dim]{trained_date}[/dim]",
            border_style="cyan",
            padding=(0, 2),
            box=box.DOUBLE
        ))

        # Step 0: Interactive Mode Selection
        console.print("\n[bold yellow]Select Mode:[/bold yellow]")
        console.print("1. [bold green]Live Mode[/bold green] (Generate signals & Update positions)")
        console.print("2. [bold magenta]Test Mode[/bold magenta] (View signals only - NO position changes)")

        mode_choice = input("\nEnter choice (1 or 2, default=2): ").strip()
        is_live = (mode_choice == '1')

        if is_live:
            console.print("[bold red]LOCKED IN LIVE MODE[/bold red]")
        else:
            console.print("[bold blue]RUNNING IN TEST MODE (VIEW ONLY)[/bold blue]")

        # Step 1: Update data
        console.print("\n[yellow]Fetching latest market data...[/yellow]")
        self._update_data()

        # Step 2: Check existing positions
        console.print("\n[yellow]Checking existing positions...[/yellow]")
        existing_positions = self._review_existing_positions()

        # Step 3: Generate new signals
        console.print("\n[yellow]Generating trading signals...[/yellow]")
        new_signals = self._generate_signals(portfolio_value, existing_positions)

        # Step 4: Display recommendations
        self._display_recommendations(existing_positions, new_signals, custom_capital=self.custom_capital)

        # Step 5: Save positions ONLY in live mode
        if new_signals and is_live:
            console.print("\n[yellow]Updating position database...[/yellow]")
            self._save_positions(new_signals)
        elif new_signals:
            console.print("\n[blue]Test Mode: Signals displayed but not saved to portfolio.[/blue]")

        return new_signals

    def _update_data(self):
        """Fetch and update latest price data."""
        console.print("\n[bold cyan]━━━ MARKET DATA UPDATE ━━━[/bold cyan]")
        
        # Check if data is already fresh (avoid redundant API calls)
        if self.storage.is_data_fresh():
            latest = self.storage.get_latest_date()
            console.print(f"  [green]✓[/green] Data already up-to-date (latest: {latest})")
            return
        
        # Fetch last 10 days to ensure we have latest data
        data = self.fetcher.fetch_batch(days=10)

        if not data.empty:
            count = self.storage.upsert_prices(data)
            console.print(f"  [green]✓[/green] Successfully updated {count} price records")
        else:
            console.print("  [yellow]⚠[/yellow] No new market data available")

    def _review_existing_positions(self) -> Dict[str, Dict]:
        """Review and provide recommendations for existing positions."""
        console.print("\n[bold cyan]━━━ PORTFOLIO REVIEW ━━━[/bold cyan]")
        positions = self.position_manager.get_open_positions()
        recommendations = {}

        if not positions:
            console.print("  [dim]No active positions found in portfolio[/dim]")
            return recommendations

        # Get latest prices
        all_data = self.storage.get_prices()

        for pos in positions:
            ticker = pos.ticker
            ticker_data = all_data[all_data['ticker'] == ticker]

            if ticker_data.empty:
                continue

            latest = ticker_data.iloc[-1]
            current_price = latest['close']
            high = latest['high']
            low = latest['low']

            entry = pos.filled_price or pos.entry_price
            pnl_pct = (current_price - entry) / entry * 100

            # Determine recommendation
            action = "HOLD"
            reason = ""

            if current_price <= pos.stop_loss:
                action = "SELL (Stop Loss Hit)"
                reason = f"Price {current_price:,.0f} <= SL {pos.stop_loss:,.0f}"
            elif current_price >= pos.take_profit:
                action = "SELL (Take Profit Hit)"
                reason = f"Price {current_price:,.0f} >= TP {pos.take_profit:,.0f}"
            elif pnl_pct < -5:
                action = "REVIEW (Large Loss)"
                reason = f"Floating loss: {pnl_pct:.1f}%"
            elif pnl_pct > 8:
                action = "CONSIDER TRAILING"
                reason = f"Floating profit: {pnl_pct:.1f}%"
            else:
                action = "HOLD"
                reason = f"P&L: {pnl_pct:+.1f}%"

            recommendations[ticker] = {
                'position': pos,
                'current_price': current_price,
                'pnl_pct': pnl_pct,
                'action': action,
                'reason': reason
            }

        return recommendations

    def _generate_signals(
        self,
        portfolio_value: float,
        existing_positions: Dict[str, Dict]
    ) -> List[Dict]:
        """Generate new trading signals."""
        # Load all data
        all_data = self.storage.get_prices()

        if all_data.empty:
            console.print("  [red]No data available. Run data fetch first.[/red]")
            return []

        # Calculate signals and ML predictions with progress bar
        data_by_ticker = {}
        ml_predictions = {}
        unique_tickers = all_data['ticker'].unique()


        # Hourly Caching Logic
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)

        # Create unique key based on date, hour, and ticker universe
        ticker_list_str = "_".join(sorted(unique_tickers))
        ticker_hash = hashlib.md5(ticker_list_str.encode()).hexdigest()[:8]
        now = datetime.now()
        current_hour = now.strftime('%Y%m%d_%H')
        cache_key = f"signals_{current_hour}_{ticker_hash}.pkl"
        cache_path = cache_dir / cache_key

        # Cleanup old caches
        self._cleanup_old_caches(cache_dir, current_hour)

        passed_tickers = []
        data_by_ticker = {}
        ml_predictions = {}
        cached_found = False

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    passed_tickers = cache_data.get('passed_tickers', [])
                    data_by_ticker = cache_data.get('data_by_ticker', {})
                    ml_predictions = cache_data.get('ml_predictions', {})
                    cached_found = True
                console.print(f"  [green]✓ Using cached signals for this hour ({cache_path.name})[/green]")
            except Exception as e:
                logger.warning(f"Failed to load signal cache: {e}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:

            if not cached_found:
                # Phase 0: Screening (PARALLEL)
                console.print(f"  [cyan]Running High-Potential Screener...[/cyan]")

                # Group data once for efficiency
                all_grouped = all_data.sort_values('date').groupby('ticker')

                screen_task = progress.add_task("Screening candidates...", total=len(unique_tickers))

                def screen_ticker(ticker):
                    """Screen a single ticker."""
                    try:
                        df = all_grouped.get_group(ticker)
                        if self.screener._check_criteria(df, ticker):
                            return ticker
                    except KeyError:
                        pass
                    return None

                # Parallel screening with ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(screen_ticker, ticker): ticker for ticker in unique_tickers}
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            passed_tickers.append(result)
                        progress.advance(screen_task)

                console.print(f"  [green]✓ {len(passed_tickers)} stocks passed screening out of {len(unique_tickers)}[/green]")

                if not passed_tickers:
                    return []

                # Phase 1: Technical signals (PARALLEL)
                sig_task = progress.add_task("Calculating technical signals...", total=len(passed_tickers))

                def calculate_signals(ticker):
                    """Calculate technical signals for a single ticker."""
                    try:
                        ticker_data = all_grouped.get_group(ticker)
                        if len(ticker_data) >= config.data.min_data_points:
                            signals = self.signal_combiner.calculate_signals(ticker_data)
                            return (ticker, signals)
                    except Exception as e:
                        logger.warning(f"Signal calculation failed for {ticker}: {e}")
                    return None

                # Parallel signal calculation
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(calculate_signals, ticker): ticker for ticker in passed_tickers}
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            ticker, signals = result
                            data_by_ticker[ticker] = signals
                        progress.advance(sig_task)

            if not passed_tickers:
                return []

            # Phase 2: ML predictions (XGBoost only) - PARALLEL & CACHED
            if self.model.model is not None:
                # Only run predictions if not cached
                if not ml_predictions:
                    ml_task = progress.add_task("Generating XGBOOST predictions...", total=len(data_by_ticker))

                    def predict_ticker(ticker_and_df):
                        """Predict for a single ticker."""
                        ticker, ticker_df = ticker_and_df
                        try:
                            _, prob = self.model.predict_latest(ticker_df)
                            return (ticker, pd.Series([prob], index=[ticker_df.index[-1]]))
                        except Exception as e:
                            logger.debug(f"XGBoost prediction failed for {ticker}: {e}")
                        return None

                    # Parallel ML predictions
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {executor.submit(predict_ticker, item): item[0]
                                   for item in data_by_ticker.items()}
                        for future in as_completed(futures):
                            result = future.result()
                            if result:
                                ticker, prediction = result
                                ml_predictions[ticker] = prediction
                            progress.advance(ml_task)

                    # Save to cache (including ML predictions)
                    if not cached_found:
                        try:
                            with open(cache_path, 'wb') as f:
                                pickle.dump({
                                    'passed_tickers': passed_tickers,
                                    'data_by_ticker': data_by_ticker,
                                    'ml_predictions': ml_predictions
                                }, f)
                        except Exception as e:
                            logger.warning(f"Failed to save signal cache: {e}")

                # Rank and process
                rankings = self.signal_combiner.rank_stocks(data_by_ticker, ml_predictions)
                if not rankings.empty:
                    # STRICT FILTERING: Must be a BUY signal AND meet the score threshold
                    buy_signals = rankings[
                        (rankings['signal'] == 'BUY') & (rankings['composite_score'] >= MIN_SIGNAL_SCORE)
                    ].copy()
                else:
                    return []
            else:
                # Indicators only fallback
                rankings = self.signal_combiner.rank_stocks(data_by_ticker, {})
                if not rankings.empty:
                    # STRICT FILTERING: Must be a BUY signal AND meet the score threshold
                    buy_signals = rankings[
                        (rankings['signal'] == 'BUY') & (rankings['composite_score'] >= MIN_SIGNAL_SCORE)
                    ].copy()
                else:
                    return []

        # PURE MODEL FOCUS: Rank strictly by strength and take top 10
        # This decouples selection from sizing/constraints
        if buy_signals is None or buy_signals.empty:
            return []

        top_candidates = buy_signals.sort_values('composite_score', ascending=False).head(10)

        # Phase 3: Final signal generation with sizing
        new_signals = []
        max_total_positions = 15
        max_new_positions = max_total_positions - len(existing_positions)

        # Get list of tickers with existing positions to avoid duplicates
        existing_tickers = set(existing_positions.keys())

        # Sector limits (still applied during sizing, but not selection)
        sector_count = {}
        max_per_sector = 3

        for _, row in top_candidates.iterrows():
            if len(new_signals) >= max_new_positions:
                break

            ticker = row['ticker']

            # Skip if ticker already has an open position
            if ticker in existing_tickers:
                logger.debug(f"Skipping {ticker} - already has open position")
                continue

            sector = self.sector_mapping.get(ticker, 'Other')

            # Determine order type based on signal strength
            score = row['composite_score']
            if score >= MARKET_ORDER_THRESHOLD:
                order_type = OrderType.MARKET.value
            else:
                order_type = OrderType.LIMIT.value

            # Get entry price and ATR
            entry_price = row['close']
            atr = row.get('atr', entry_price * 0.02)

            # For limit orders, set limit price slightly below current
            if order_type == OrderType.LIMIT.value:
                # Limit price at lower band or 1-2% below current
                limit_discount = min(0.02, atr / entry_price)
                limit_price = entry_price * (1 - limit_discount)
            else:
                limit_price = None

            # Determine strategy mode & sizing boost
            strategy_mode = row.get('strategy_mode', 'BASELINE')
            sizing_boost = 1.618 if strategy_mode == 'EXPLOSIVE' else 1.0 # Golden ratio boost for snipers

            # Calculate exit levels
            exit_levels = self.exit_manager.calculate_levels(entry_price, atr, mode=strategy_mode)

            # Calculate position size (Fixed bug: pass available capital, not pre-divided)
            try:
                size_info = self.position_sizer.calculate_position_size(
                    portfolio_value=portfolio_value,
                    stock_volatility=row.get('volatility_20', 0.3),
                    avg_market_volatility=0.2,
                    entry_price=entry_price,
                    stop_loss=exit_levels.stop_loss,
                    confidence=min(1.0, score + 0.5)  # Offset to ensure meaningful size for buy signals
                )
            except Exception as e:
                logger.debug(f"Position sizing failed for {ticker}: {e}")
                continue

            if size_info['shares'] <= 0:
                logger.debug(f"Skipping {ticker} due to 0 shares (Size: {size_info['position_value']:.0f})")
                continue

            signal = {
                'ticker': ticker,
                'sector': sector,
                'score': score,
                'strategy_mode': strategy_mode,
                'order_type': order_type,
                'entry_price': round(entry_price, 0),
                'limit_price': round(limit_price, 0) if limit_price else None,
                'stop_loss': round(exit_levels.stop_loss, 0),
                'take_profit': round(exit_levels.take_profit, 0),
                'shares': int(size_info['shares'] * sizing_boost),
                'position_value': round(size_info['position_value'] * sizing_boost, 0),
                'risk_reward': exit_levels.reward_ratio
            }

            new_signals.append(signal)
            sector_count[sector] = sector_count.get(sector, 0) + 1

        return new_signals

    def _display_recommendations(
        self,
        existing_positions: Dict[str, Dict],
        new_signals: List[Dict],
        custom_capital: float = 0.0
    ):
        """Display recommendations in formatted tables."""
        # Existing Positions Table
        if existing_positions:
            console.print("\n[bold]EXISTING POSITIONS[/bold]")

            table = Table(box=box.ROUNDED)
            table.add_column("Ticker", style="cyan bold")
            table.add_column("Sector", style="dim")
            table.add_column("Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("P&L", justify="right")
            table.add_column("Recommendation", justify="center")

            for ticker, data in existing_positions.items():
                pos = data['position']
                entry = pos.filled_price or pos.entry_price
                pnl_style = "bold green" if data['pnl_pct'] > 0 else "bold red"

                # Use symbols for quick identification
                action_color = "red" if "SELL" in data['action'] else "yellow" if "REVIEW" in data['action'] else "green"

                table.add_row(
                    ticker.replace('.JK', ''),
                    self.sector_mapping.get(ticker, 'Unknown'),
                    f"Rp {entry:,.0f}",
                    f"Rp {data['current_price']:,.0f}",
                    f"[{pnl_style}]{data['pnl_pct']:+.2f}%[/{pnl_style}]",
                    f"[{action_color}]{data['action']}[/{action_color}]"
                )

            console.print(table)

        if not new_signals:
            console.print("\n[bold yellow]NEW TRADING SIGNALS[/bold yellow]")
            console.print("  No new signals meet criteria")
            return

        if new_signals:
            console.print("\n[bold cyan]━━━ NEW TRADING OPPORTUNITIES ━━━[/bold cyan]")
            self._print_signal_table(new_signals, custom_capital=custom_capital)

        # Summary Footer
        total_value = sum(s['position_value'] for s in new_signals)
        market_orders = sum(1 for s in new_signals if s['order_type'] == 'MARKET')
        limit_orders = len(new_signals) - market_orders

        console.print("\n" + "─" * 60)
        summary_text = (
            f"Total Opportunities: [bold white]{len(new_signals)}[/bold white] | "
            f"Market Orders: [bold green]{market_orders}[/bold green] | "
            f"Limit Orders: [bold yellow]{limit_orders}[/bold yellow]\n"
            f"Estimated Portfolio Exposure: [bold cyan]Rp {total_value:,.0f}[/bold cyan]"
        )

        if custom_capital > 0:
            summary_text += f"\nCustom Capital to Allocate: [bold green]Rp {custom_capital:,.0f}[/bold green]"

        console.print(Panel(summary_text, border_style="dim", padding=(0, 2)))
        console.print("[dim]⚠️  Execute based on pre-market opening conditions. Check for major gap-ups/downs.[/dim]")
        console.print("[dim]📅 Maximum hold period: 5 trading days. Close positions automatically at day 5 or when TP/SL is hit.[/dim]")
        console.print()

    def _print_signal_table(self, signals: List[Dict], custom_capital: float = 0.0):
        """Helper to print a rich table for a group of signals."""
        table = Table(box=box.SIMPLE_HEAD, show_lines=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Ticker", style="cyan bold")
        table.add_column("Mode", justify="center")
        table.add_column("Execution", justify="center")
        table.add_column("Strength", justify="right", style="magenta")
        table.add_column("Price", justify="right")
        table.add_column("Limit", justify="right")
        table.add_column("Stop Loss", justify="right", style="red")
        table.add_column("Target", justify="right", style="green")
        table.add_column("Allocation", justify="right")

        if custom_capital > 0:
            table.add_column("Custom", justify="right", style="bold green")

        # Calculate total allocation to derive relative weights
        total_alloc = sum(sig['position_value'] for sig in signals)

        for i, sig in enumerate(signals, 1):
            order_style = "bold green" if sig['order_type'] == 'MARKET' else "yellow"
            conf_pct = min(100, max(0, sig['score'] * 100))

            mode_icon = "🔥 [bold red]SNIPER[/bold red]" if sig.get('strategy_mode') == 'EXPLOSIVE' else "[dim]⚖️ BASE[/dim]"

            # Calculate percentages relative to entry/limit price
            reference_price = sig['limit_price'] if sig['limit_price'] else sig['entry_price']
            sl_pct = ((sig['stop_loss'] - reference_price) / reference_price) * 100
            tp_pct = ((sig['take_profit'] - reference_price) / reference_price) * 100

            row_data = [
                str(i),
                sig['ticker'].replace('.JK', ''),
                mode_icon,
                f"[{order_style}]{sig['order_type']}[/{order_style}]",
                f"{conf_pct:.1f}%",
                f"Rp {sig['entry_price']:,.0f}",
                f"Rp {sig['limit_price']:,.0f}" if sig['limit_price'] else "-",
                f"{sig['stop_loss']:,.0f} ({sl_pct:+.1f}%)",
                f"{sig['take_profit']:,.0f} (+{tp_pct:.1f}%)",
                f"Rp {sig['position_value']:,.0f}"
            ]

            if custom_capital > 0:
                share = sig['position_value'] / total_alloc if total_alloc > 0 else 0
                custom_amt = share * custom_capital
                row_data.append(f"Rp {custom_amt:,.0f}")

            table.add_row(*row_data)

        console.print(table)

    def _save_positions(self, signals: List[Dict]):
        """Save new positions to database."""
        today = date.today().isoformat()

        for sig in signals:
            position = Position(
                ticker=sig['ticker'],
                order_type=sig['order_type'],
                entry_price=sig['entry_price'],
                limit_price=sig['limit_price'],
                stop_loss=sig['stop_loss'],
                take_profit=sig['take_profit'],
                shares=sig['shares'],
                position_value=sig['position_value'],
                signal_score=sig['score'],
                strategy_mode=sig.get('strategy_mode', 'BASELINE'),
                created_date=today,
                status=PositionStatus.PENDING.value if sig['order_type'] == 'LIMIT' else PositionStatus.OPEN.value,
                filled_date=today if sig['order_type'] == 'MARKET' else None,
                filled_price=sig['entry_price'] if sig['order_type'] == 'MARKET' else None
            )

            self.position_manager.create_position(position)

        console.print(f"\n[green]✓ Saved {len(signals)} positions[/green]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='IHSG Morning Trading Signals')
    parser.add_argument('--custom-capital', type=float, default=0.0, help='Custom capital to distribute among signals')
    args = parser.parse_args()

    signals = MorningSignals()
    signals.run(custom_capital=args.custom_capital)


if __name__ == "__main__":
    main()
