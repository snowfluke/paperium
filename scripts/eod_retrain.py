#!/usr/bin/env python3
"""
End-of-Day Retraining Script
Run after market close to update positions and retrain the model

Usage:
    uv run python scripts/eod_retrain.py
    
Or if installed:
    uv run eod
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from datetime import datetime, date
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import config
from data.fetcher import DataFetcher
from data.storage import DataStorage
from signals.combiner import SignalCombiner
from ml.model import TradingModel
from strategy.position_manager import PositionManager, PositionStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()


class EODRetraining:
    """
    End-of-day position evaluation and model retraining.
    
    Flow:
    1. Fetch latest EOD data
    2. Evaluate all open positions against today's price action
    3. Update position statuses (hit SL/TP, floating P&L, etc.)
    4. Expire unfilled limit orders
    5. Retrain ML model with new data
    6. Show performance summary
    """
    
    def __init__(self):
        self.storage = DataStorage(config.data.db_path)
        self.fetcher = DataFetcher(config.data.stock_universe)
        self.signal_combiner = SignalCombiner(config)
        self.position_manager = PositionManager()
    
    def run(self):
        """Run the EOD retraining pipeline."""
        console.print(Panel.fit(
            f"[bold magenta]IHSG End-of-Day Processing[/bold magenta]\n"
            f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="magenta"
        ))
        
        # Step 1: Fetch latest data
        console.print("\n[yellow]Fetching end-of-day data...[/yellow]")
        self._fetch_eod_data()
        
        # Step 2: Evaluate positions
        console.print("\n[yellow]Evaluating positions...[/yellow]")
        results = self._evaluate_positions()
        
        # Step 3: Expire unfilled orders
        console.print("\n[yellow]⏰ Processing unfilled orders...[/yellow]")
        expired = self._expire_unfilled_orders()
        
        # Step 4: Retrain model
        console.print("\n[yellow]Retraining ML model...[/yellow]")
        metrics = self._retrain_model()
        
        # Step 5: Display summary
        self._display_summary(results, expired, metrics)
        
        return results
    
    def _fetch_eod_data(self):
        """Fetch latest EOD prices."""
        data = self.fetcher.fetch_batch(days=5)
        
        if not data.empty:
            count = self.storage.upsert_prices(data)
            console.print(f"  ✓ Updated {count} EOD records")
        else:
            console.print("  ⚠ No EOD data available")
    
    def _evaluate_positions(self) -> Dict:
        """Evaluate all open positions against today's price action."""
        positions = self.position_manager.get_open_positions()
        all_data = self.storage.get_prices()

        results = {
            'hit_tp': [],
            'hit_sl': [],
            'max_hold_closed': [],
            'floating_profit': [],
            'floating_loss': [],
            'filled': [],
            'unchanged': []
        }

        if not positions:
            console.print("  No open positions to evaluate")
            return results

        today = date.today()

        for pos in positions:
            ticker = pos.ticker
            ticker_data = all_data[all_data['ticker'] == ticker]

            if ticker_data.empty:
                continue

            latest = ticker_data.iloc[-1]
            current_price = latest['close']
            high = latest['high']
            low = latest['low']

            # Check max hold period (5 trading days from filled_date)
            # Only check for positions that have been filled (not PENDING)
            if pos.status in ('OPEN', 'FLOATING_PROFIT', 'FLOATING_LOSS') and pos.filled_date:
                filled_date = datetime.strptime(pos.filled_date, '%Y-%m-%d').date()
                days_held = (today - filled_date).days

                if days_held >= 5:
                    # Close position due to max hold period
                    entry = pos.filled_price or pos.entry_price
                    result = self.position_manager.close_position(
                        ticker, current_price, "MAX_HOLD_PERIOD"
                    )

                    if result:
                        result_entry = {
                            'ticker': ticker,
                            'entry': entry,
                            'current': current_price,
                            'pnl_pct': result['pnl_pct'],
                            'days_held': days_held,
                            'stop_loss': pos.stop_loss,
                            'take_profit': pos.take_profit
                        }
                        results['max_hold_closed'].append(result_entry)
                        continue  # Skip normal update since we just closed it

            # Update position status (check TP/SL hits)
            update = self.position_manager.update_position_status(
                ticker, current_price, high, low
            )

            if not update:
                continue

            entry = pos.filled_price or pos.entry_price
            pnl_pct = (current_price - entry) / entry * 100

            result_entry = {
                'ticker': ticker,
                'entry': entry,
                'current': current_price,
                'pnl_pct': pnl_pct,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit
            }

            if update.get('exit'):
                if update['reason'] == 'HIT_TAKE_PROFIT':
                    results['hit_tp'].append(result_entry)
                elif update['reason'] == 'HIT_STOP_LOSS':
                    results['hit_sl'].append(result_entry)
            elif update.get('filled'):
                results['filled'].append(result_entry)
            elif pnl_pct > 0:
                results['floating_profit'].append(result_entry)
            elif pnl_pct < 0:
                results['floating_loss'].append(result_entry)
            else:
                results['unchanged'].append(result_entry)

        return results
    
    def _expire_unfilled_orders(self) -> int:
        """Expire limit orders that weren't filled today."""
        today = date.today().isoformat()
        expired = self.position_manager.expire_unfilled_orders(today)
        
        if expired > 0:
            console.print(f"  ⏰ Expired {expired} unfilled limit orders")
        else:
            console.print("  ✓ No orders to expire")
        
        return expired
    
    def _retrain_model(self) -> Dict:
        """Retrain global XGBoost model with pooled data from all tickers."""
        all_data = self.storage.get_prices()
        
        if all_data.empty:
            console.print("  ⚠ No data available for training")
            return {}
        
        # Pool data from all tickers with progress bar
        pooled_data = []
        ticker_count = 0
        unique_tickers = all_data['ticker'].unique()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            pool_task = progress.add_task("Pooling price data...", total=len(unique_tickers))
            for ticker in unique_tickers:
                ticker_data = all_data[all_data['ticker'] == ticker].copy()
                ticker_data = ticker_data.sort_values('date')
                
                if len(ticker_data) >= 60:
                    pooled_data.append(ticker_data)
                    ticker_count += 1
                progress.advance(pool_task)
            
            if not pooled_data:
                console.print("  ⚠ Insufficient data for global training")
                return {}
                
            full_df = pd.concat(pooled_data)
            
            # Retrain XGBoost
            self.metrics_summary = {'trained_count': ticker_count}
            
            progress.add_task("Retraining XGBOOST...", total=None)
            challenger = TradingModel(config.ml)
            # Warm start: load current champion if available
            champ_path = os.path.join('models', 'global_xgb_champion.pkl')
            if os.path.exists(champ_path):
                try:
                    challenger.load(champ_path)
                    metrics = challenger.train(full_df, base_model=challenger.model)
                except:
                    metrics = challenger.train(full_df)
            else:
                metrics = challenger.train(full_df)
            
            acc_challenger = metrics.get('cv_accuracy', 0)
            
            if acc_challenger > 0.60:
                console.print(f"  [green]✓ XGBOOST passed validation ({acc_challenger:.1%})[/green]")
                challenger.save(os.path.join('models', 'global_xgb_champion.pkl'))
                self.metrics_summary['xgboost'] = {'status': 'UPDATED', 'accuracy': acc_challenger}
            else:
                # Load current champion accuracy from metadata
                current_best = acc_challenger
                metadata_path = 'models/champion_metadata.json'
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            current_best = metadata.get('xgboost', {}).get('win_rate', acc_challenger)
                    except:
                        pass
                console.print(f"  [red]✕ XGBOOST failed validation ({acc_challenger:.1%})[/red]")
                self.metrics_summary['xgboost'] = {'status': 'KEPT_OLD', 'accuracy': current_best}
            
            progress.stop()

        return self.metrics_summary
    
    def _display_summary(self, results: Dict, expired: int, metrics: Dict):
        """Display EOD summary."""
        console.print("\n" + "=" * 60)
        console.print("[bold]END-OF-DAY SUMMARY[/bold]")
        console.print("=" * 60)

        # Position outcomes
        hit_tp = results.get('hit_tp', [])
        hit_sl = results.get('hit_sl', [])
        max_hold_closed = results.get('max_hold_closed', [])
        floating_profit = results.get('floating_profit', [])
        floating_loss = results.get('floating_loss', [])
        filled = results.get('filled', [])

        # Take Profits
        if hit_tp:
            console.print("\n[bold green]✅ TAKE PROFIT HIT[/bold green]")
            for item in hit_tp:
                console.print(f"  {item['ticker']}: Entry {item['entry']:,.0f} → "
                             f"TP {item['take_profit']:,.0f} "
                             f"([green]+{item['pnl_pct']:.1f}%[/green])")

        # Stop Losses
        if hit_sl:
            console.print("\n[bold red]❌ STOP LOSS HIT[/bold red]")
            for item in hit_sl:
                console.print(f"  {item['ticker']}: Entry {item['entry']:,.0f} → "
                             f"SL {item['stop_loss']:,.0f} "
                             f"([red]{item['pnl_pct']:.1f}%[/red])")

        # Max Hold Period Closed
        if max_hold_closed:
            console.print("\n[bold yellow]📅 MAX HOLD PERIOD (5 DAYS)[/bold yellow]")
            for item in max_hold_closed:
                pnl_color = "green" if item['pnl_pct'] > 0 else "red"
                pnl_sign = "+" if item['pnl_pct'] > 0 else ""
                console.print(f"  {item['ticker']}: Held {item['days_held']} days → "
                             f"Exit {item['current']:,.0f} "
                             f"([{pnl_color}]{pnl_sign}{item['pnl_pct']:.1f}%[/{pnl_color}])")

        # Filled Orders
        if filled:
            console.print("\n[bold cyan]📥 LIMIT ORDERS FILLED[/bold cyan]")
            for item in filled:
                console.print(f"  {item['ticker']}: Filled at {item['entry']:,.0f}")
        
        # Floating P&L Summary
        total_floating_profit = sum(p['pnl_pct'] for p in floating_profit)
        total_floating_loss = sum(p['pnl_pct'] for p in floating_loss)
        
        console.print(f"\n[bold]FLOATING POSITIONS[/bold]")
        console.print(f"  Profitable: {len(floating_profit)} positions "
                     f"([green]+{total_floating_profit:.1f}%[/green])")
        console.print(f"  Losing:     {len(floating_loss)} positions "
                     f"([red]{total_floating_loss:.1f}%[/red])")
        
        # Daily Statistics
        console.print(f"\n[bold]TODAY'S STATISTICS[/bold]")
        console.print(f"  Take Profits:     {len(hit_tp)}")
        console.print(f"  Stop Losses:      {len(hit_sl)}")
        console.print(f"  Max Hold Closed:  {len(max_hold_closed)}")
        console.print(f"  Orders Filled:    {len(filled)}")
        console.print(f"  Orders Expired:   {expired}")

        # Win rate for closed positions today
        closed_today = len(hit_tp) + len(hit_sl) + len(max_hold_closed)
        wins_today = len(hit_tp) + len([p for p in max_hold_closed if p['pnl_pct'] > 0])
        if closed_today > 0:
            win_rate = wins_today / closed_today * 100
            console.print(f"  Today's Win Rate: {win_rate:.0f}%")
        
        # Model metrics
        if metrics:
            console.print(f"\n[bold]MODEL RETRAINING[/bold]")
            console.print(f"  Stocks trained:   {metrics.get('trained_count', 0)}")
            if 'xgboost' in metrics:
                m_data = metrics['xgboost']
                style = "green" if m_data['status'] == 'UPDATED' else "yellow"
                console.print(f"  XGBOOST: [{style}]{m_data['status']}[/{style}] "
                             f"(Acc: {m_data['accuracy']:.1%})")
        
        # Overall performance
        perf = self.position_manager.get_performance_summary()
        if perf['total_trades'] > 0:
            console.print(f"\n[bold]OVERALL PERFORMANCE[/bold]")
            console.print(f"  Total Trades:     {perf['total_trades']}")
            console.print(f"  Win Rate:         {perf['win_rate']:.1f}%")
            console.print(f"  Avg Win:          +{perf['avg_win_pct']:.1f}%")
            console.print(f"  Avg Loss:         {perf['avg_loss_pct']:.1f}%")
            console.print(f"  Total P&L:        Rp {perf['total_pnl']:,.0f}")
        
        console.print("\n" + "=" * 60)
        console.print("[dim]EOD processing complete[/dim]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='IHSG End-of-Day Processing')
    parser.add_argument('--skip-retrain', action='store_true',
                       help='Skip model retraining')
    
    args = parser.parse_args()
    
    eod = EODRetraining()
    
    if args.skip_retrain:
        console.print("[yellow]Skipping model retraining[/yellow]")
        # Only evaluate positions
        eod._fetch_eod_data()
        eod._evaluate_positions()
        eod._expire_unfilled_orders()
    else:
        eod.run()


if __name__ == "__main__":
    main()
