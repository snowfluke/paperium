#!/usr/bin/env python3
"""
Ensemble Backtest (LSTM + XGBoost)
Walk-forward backtesting using combined LSTM and XGBoost predictions.
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from ml.model import ModelWrapper
from ml.features import FeatureEngineer
from ml.xgb_inference import XGBoostInference
from utils.logger import TimedLogger
from data.blacklist import BLACKLIST_UNIVERSE

console = Console()


class EnsembleBacktest:
    """Backtesting engine for LSTM + XGBoost ensemble."""

    def __init__(self, start_date, end_date):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = 100_000_000  # 100M IDR
        self.capital = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.max_positions = 8
        self.position_size_pct = 0.125  # 12.5% per position

        # Load models
        self.logger = TimedLogger()
        self.lstm_model = None
        self.xgb_model = None
        self.feature_eng = FeatureEngineer(config)

        # Database connection
        self.db_path = config.data.db_path

    def load_models(self):
        """Load LSTM and XGBoost models."""
        # Load LSTM
        self.logger.log("[cyan]Loading LSTM model...[/cyan]")
        self.lstm_model = ModelWrapper(config)
        epoch, metrics = self.lstm_model.load_checkpoint("models/best_lstm.pt")
        if epoch is None:
            console.print("[red]Failed to load LSTM model[/red]")
            return False
        self.logger.log(f"[green]✓ LSTM loaded:[/green] Epoch {epoch}, Val Acc: {metrics.get('val_acc', 0):.1%}")

        # Load XGBoost
        self.logger.log("[cyan]Loading XGBoost model...[/cyan]")
        self.xgb_model = XGBoostInference()
        if not self.xgb_model.load_model():
            console.print("[red]XGBoost model not found. Cannot run ensemble backtest.[/red]")
            console.print("[yellow]Please train XGBoost model in paperium-v1 branch first.[/yellow]")
            return False
        self.logger.log("[green]✓ XGBoost loaded[/green]")

        return True

    def get_trading_dates(self):
        """Get list of trading dates in date range."""
        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT DISTINCT date
        FROM prices
        WHERE date >= '{self.start_date.strftime('%Y-%m-%d')}'
          AND date <= '{self.end_date.strftime('%Y-%m-%d')}'
        ORDER BY date
        """
        dates = pd.read_sql_query(query, conn)['date'].tolist()
        conn.close()
        return dates

    def load_ticker_history(self, ticker, end_date, limit=200):
        """Load historical data for ticker up to end_date."""
        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT date, open, high, low, close, volume
        FROM prices
        WHERE ticker = '{ticker}'
          AND date <= '{end_date}'
        ORDER BY date DESC
        LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.iloc[::-1] if not df.empty else None

    def load_all_tickers_batch(self, tickers, end_date, limit=200):
        """Batch load historical data for multiple tickers."""
        from data.universe import IDX_UNIVERSE

        conn = sqlite3.connect(self.db_path)

        # Build query for all tickers at once
        ticker_list = "','".join(tickers)
        query = f"""
        WITH RankedPrices AS (
            SELECT
                ticker,
                date,
                open,
                high,
                low,
                close,
                volume,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
            FROM prices
            WHERE ticker IN ('{ticker_list}')
              AND date <= '{end_date}'
        )
        SELECT ticker, date, open, high, low, close, volume
        FROM RankedPrices
        WHERE rn <= {limit}
        ORDER BY ticker, date
        """

        df_all = pd.read_sql_query(query, conn)
        conn.close()

        # Split into dict by ticker
        result = {}
        for ticker in tickers:
            ticker_df = df_all[df_all['ticker'] == ticker].copy()
            if len(ticker_df) >= config.data.window_size:
                ticker_df = ticker_df.drop(columns=['ticker'])
                result[ticker] = ticker_df

        return result

    def get_future_prices(self, ticker, start_date, days=10):
        """Get future prices for exit simulation."""
        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT date, open, high, low, close
        FROM prices
        WHERE ticker = '{ticker}'
          AND date > '{start_date}'
        ORDER BY date
        LIMIT {days}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def scan_signals(self, current_date):
        """Scan for buy signals using ensemble approach."""
        from data.universe import IDX_UNIVERSE
        signals = []

        # Filter out blacklisted and already-held tickers
        held_tickers = {pos['ticker'] for pos in self.positions}
        tickers_to_scan = [t for t in IDX_UNIVERSE if t not in BLACKLIST_UNIVERSE and t not in held_tickers]

        # Batch load all ticker data at once
        ticker_data = self.load_all_tickers_batch(tickers_to_scan, current_date, limit=config.data.window_size + 10)

        for ticker, df in ticker_data.items():
            try:
                # LSTM prediction
                X_tensor = self.feature_eng.prepare_inference(df)
                pred_class, probs = self.lstm_model.predict(X_tensor)
                pred_class = pred_class[0]
                probs = probs[0]
                lstm_prob = probs[2]

                # Filter: LSTM must predict Class 2 with >50% confidence
                if pred_class != 2 or lstm_prob <= 0.5:
                    continue

                # XGBoost prediction (only if LSTM passed)
                xgb_result = self.xgb_model.predict(df)
                if xgb_result is None:
                    continue

                xgb_conf = xgb_result['confidence']
                xgb_threshold = self.xgb_model.get_threshold()

                # Filter: XGBoost confidence must exceed threshold
                if xgb_conf < xgb_threshold:
                    continue

                # Combined score
                combined_score = (lstm_prob + xgb_conf) / 2.0

                latest = df.iloc[-1]
                price = latest['close']

                signals.append({
                    'ticker': ticker,
                    'price': price,
                    'lstm_conf': lstm_prob,
                    'xgb_conf': xgb_conf,
                    'combined_score': combined_score,
                    'sl_pct': xgb_result['sl_pct'],
                    'tp_pct': xgb_result['tp_pct'],
                    'trail_pct': xgb_result['trail_pct'],
                    'entry_pct': xgb_result['entry_pct'],
                    'order_type': xgb_result['order_type']
                })

            except Exception:
                continue

        # Sort by combined score
        signals.sort(key=lambda x: x['combined_score'], reverse=True)
        return signals

    def open_position(self, signal, entry_date):
        """Open a new position."""
        # Calculate entry price (limit order: below current price)
        entry_price = signal['price'] * (1 - signal['entry_pct'])

        # Calculate position size using entry price
        available_capital = self.capital - sum(p['entry_price'] * p['shares'] for p in self.positions)
        position_value = available_capital * self.position_size_pct

        shares = int(position_value / entry_price / 100) * 100
        if shares == 0:
            return False

        actual_value = shares * entry_price

        # Calculate SL/TP/Trail prices from entry price
        sl_price = entry_price * (1 - signal['sl_pct'])
        tp_price = entry_price * (1 + signal['tp_pct'])
        trail_price = entry_price * (1 - signal['trail_pct'])

        position = {
            'ticker': signal['ticker'],
            'entry_date': entry_date,
            'entry_price': entry_price,
            'shares': shares,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'trail_price': trail_price,
            'current_trail': trail_price,
            'highest_price': entry_price,
            'lstm_conf': signal['lstm_conf'],
            'xgb_conf': signal['xgb_conf'],
            'combined_score': signal['combined_score'],
            'hold_days': 0,
            'order_type': signal['order_type']
        }

        self.positions.append(position)
        return True

    def update_positions(self, current_date):
        """Update open positions and check for exits."""
        closed_today = []

        for pos in self.positions[:]:
            # Get today's price action
            future_df = self.get_future_prices(pos['ticker'], pos['entry_date'], days=1)
            if future_df.empty:
                continue

            today = future_df.iloc[0]
            pos['hold_days'] += 1

            # Update trailing stop
            if today['high'] > pos['highest_price']:
                pos['highest_price'] = today['high']
                new_trail = pos['highest_price'] * (1 - (pos['trail_price'] / pos['entry_price'] - 1))
                pos['current_trail'] = max(pos['current_trail'], new_trail)

            # Check exit conditions
            exit_reason = None
            exit_price = None

            # 1. Stop Loss
            if today['low'] <= pos['sl_price']:
                exit_reason = 'SL'
                exit_price = pos['sl_price']

            # 2. Take Profit
            elif today['high'] >= pos['tp_price']:
                exit_reason = 'TP'
                exit_price = pos['tp_price']

            # 3. Trailing Stop
            elif today['low'] <= pos['current_trail']:
                exit_reason = 'TRAIL'
                exit_price = pos['current_trail']

            # 4. Time Stop (5 days)
            elif pos['hold_days'] >= 5:
                exit_reason = 'TIME'
                exit_price = today['close']

            if exit_reason:
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                pnl_pct = (exit_price / pos['entry_price'] - 1)

                trade = {
                    'ticker': pos['ticker'],
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_days': pos['hold_days'],
                    'exit_reason': exit_reason,
                    'lstm_conf': pos['lstm_conf'],
                    'xgb_conf': pos['xgb_conf'],
                    'combined_score': pos['combined_score'],
                    'order_type': pos['order_type']
                }

                self.closed_trades.append(trade)
                closed_today.append(pos)
                self.capital += pnl

        # Remove closed positions
        for pos in closed_today:
            self.positions.remove(pos)

    def run(self):
        """Run the backtest."""
        if not self.load_models():
            return None

        console.print(Panel.fit(
            "[bold magenta]Ensemble Backtest (LSTM + XGBoost)[/bold magenta]\n"
            f"[dim]{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}[/dim]",
            border_style="magenta"
        ))

        trading_dates = self.get_trading_dates()
        self.logger.log(f"[cyan]Found {len(trading_dates)} trading days[/cyan]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[cyan]{task.fields[positions_info]}"),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Backtesting...",
                total=len(trading_dates),
                positions_info="Positions: 0 | Trades: 0"
            )

            for i, current_date in enumerate(trading_dates):
                # Update existing positions
                self.update_positions(current_date)

                # Scan for new signals if we have capacity
                if len(self.positions) < self.max_positions:
                    signals = self.scan_signals(current_date)

                    # Open positions for top signals
                    for signal in signals[:self.max_positions - len(self.positions)]:
                        self.open_position(signal, current_date)

                # Update progress with stats
                progress.update(
                    task,
                    description=f"[cyan]{current_date}",
                    positions_info=f"Positions: {len(self.positions)} | Trades: {len(self.closed_trades)}",
                    advance=1
                )

        # Close any remaining positions at final price
        for pos in self.positions[:]:
            future_df = self.get_future_prices(pos['ticker'], pos['entry_date'], days=100)
            if not future_df.empty:
                final_price = future_df.iloc[-1]['close']
                pnl = (final_price - pos['entry_price']) * pos['shares']
                pnl_pct = (final_price / pos['entry_price'] - 1)

                trade = {
                    'ticker': pos['ticker'],
                    'entry_date': pos['entry_date'],
                    'exit_date': self.end_date.strftime('%Y-%m-%d'),
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_days': pos['hold_days'],
                    'exit_reason': 'EOD',
                    'lstm_conf': pos['lstm_conf'],
                    'xgb_conf': pos['xgb_conf'],
                    'combined_score': pos['combined_score'],
                    'order_type': pos['order_type']
                }
                self.closed_trades.append(trade)

        return self.generate_report()

    def generate_report(self):
        """Generate performance report."""
        if not self.closed_trades:
            console.print("\n[yellow]No trades executed during backtest period[/yellow]")
            return None

        df = pd.DataFrame(self.closed_trades)

        # Calculate metrics
        total_trades = len(df)
        wins = len(df[df['pnl'] > 0])
        losses = len(df[df['pnl'] < 0])
        win_rate = wins / total_trades if total_trades > 0 else 0

        avg_win = df[df['pnl'] > 0]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl_pct'].mean() if losses > 0 else 0
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        total_pnl = df['pnl'].sum()
        total_return_pct = (self.capital / self.initial_capital - 1) * 100

        avg_pnl_per_trade = df['pnl_pct'].mean() * 100
        median_pnl = df['pnl_pct'].median() * 100

        # Exit reasons breakdown
        exit_counts = df['exit_reason'].value_counts()

        # Display metrics
        console.print("\n")
        console.print(Panel.fit("[bold cyan]Performance Summary[/bold cyan]", border_style="cyan"))

        metrics_table = Table(show_header=False, box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="bold white")

        metrics_table.add_row("Total Trades", f"{total_trades}")
        metrics_table.add_row("Win Rate", f"{win_rate:.1%}")
        metrics_table.add_row("W/L Ratio", f"{wl_ratio:.2f}x")
        metrics_table.add_row("Total Return", f"{total_return_pct:.2f}%")
        metrics_table.add_row("Total P/L", f"Rp {total_pnl:,.0f}")
        metrics_table.add_row("Avg P/L per Trade", f"{avg_pnl_per_trade:.2f}%")
        metrics_table.add_row("Median P/L", f"{median_pnl:.2f}%")

        console.print(metrics_table)

        # Exit reasons
        console.print("\n[bold]Exit Reasons:[/bold]")
        for reason, count in exit_counts.items():
            pct = count / total_trades * 100
            console.print(f"  {reason}: {count} ({pct:.1f}%)")

        # Order type breakdown
        console.print("\n[bold]Order Types:[/bold]")
        order_counts = df['order_type'].value_counts()
        for order_type, count in order_counts.items():
            avg_return = df[df['order_type'] == order_type]['pnl_pct'].mean() * 100
            console.print(f"  {order_type}: {count} trades (avg: {avg_return:.2f}%)")

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'wl_ratio': wl_ratio,
            'total_return_pct': total_return_pct,
            'avg_pnl_per_trade': avg_pnl_per_trade
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Ensemble Backtest')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-09-30', help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    backtest = EnsembleBacktest(args.start, args.end)
    backtest.run()


if __name__ == "__main__":
    main()
