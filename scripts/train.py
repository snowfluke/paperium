#!/usr/bin/env python3
"""
Progressive Model Training
Shows incremental improvement like training YOLO - each epoch gets better.
"""
import sys
import os
import argparse
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.live import Live
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from scripts.eval import MLBacktest
from ml.features import FeatureEngineer

console = Console()

def main():
    parser = argparse.ArgumentParser(description='Progressive Model Training')
    parser.add_argument('--days', type=str, default='max', help='Evaluation period')
    parser.add_argument('--train-window', type=str, default='max', help='Training window')
    parser.add_argument('--max-depth', type=int, default=5, help='XGBoost max depth')
    parser.add_argument('--n-estimators', type=int, default=50, help='Final n_estimators (50 = sweet spot)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (5 = 10 trees/epoch)')

    args = parser.parse_args()

    # Update config
    config.ml.use_gpu = args.gpu
    config.ml.max_depth = args.max_depth
    config.ml.n_estimators = args.n_estimators

    # Process dates
    if args.days == 'max' or args.train_window == 'max':
        conn = sqlite3.connect(config.data.db_path)
        df_dates = pd.read_sql("SELECT MIN(date), MAX(date) FROM prices", conn)
        conn.close()

        if args.days == 'max':
            eval_days = 365
            console.print(f"[dim]Auto-setting eval days to {eval_days} (1 year)[/dim]")
        else:
            eval_days = int(args.days)

        if args.train_window == 'max':
            train_window = 756  # 3 years
            console.print(f"[dim]Auto-setting train window to {train_window} days (3 years)[/dim]")
        else:
            train_window = int(args.train_window)
    else:
        eval_days = int(args.days)
        train_window = int(args.train_window)

    console.print(f"[bold cyan]Progressive Model Training[/bold cyan]")
    console.print(f"[dim]Like training YOLO - watch the model improve each epoch[/dim]\n")

    # Session setup - create dedicated folder for this training run
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = f"models/training_{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    session_file = f"{session_dir}/session.json"
    session_data = {
        "session_id": session_id,
        "session_dir": session_dir,
        "start_time": datetime.now().isoformat(),
        "parameters": {
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "epochs": args.epochs,
            "eval_days": eval_days,
            "train_window": train_window,
            "use_gpu": args.gpu,
            "sl_atr_mult": config.exit.stop_loss_atr_mult,
            "tp_atr_mult": config.exit.take_profit_atr_mult,
            "signal_threshold": config.exit.signal_threshold
        },
        "epochs": []
    }

    console.print(f"[dim]Session folder: {session_dir}[/dim]")

    # Feature set
    feature_eng = FeatureEngineer(config.ml)
    feature_count = len(feature_eng.feature_set)

    console.print(f"[bold green]Universal Feature Set ({feature_count} features)[/bold green]")
    console.print("  Base 46 technical indicators")
    console.print("  5 intraday behavior proxies\n")

    # Config display
    console.print(f"[bold]Configuration:[/bold]")
    console.print(f"  SL/TP: {config.exit.stop_loss_atr_mult:.1f}x / {config.exit.take_profit_atr_mult:.1f}x ATR")
    console.print(f"  Signal Threshold: {config.exit.signal_threshold:.2f}")
    console.print(f"  Max Depth: {args.max_depth}")
    console.print(f"  Total Estimators: {args.n_estimators}")
    console.print(f"  Training Epochs: {args.epochs}\n")

    # GPU warning
    if args.gpu:
        import sys as sys_module
        if sys_module.platform == "darwin":
            console.print("[yellow]XGBoost MPS (Metal) can be unstable. Using CPU 'hist' instead.[/yellow]\n")

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    if args.days == 'max':
        current_year = datetime.now().year
        start_date = f"{current_year - 1}-12-01"
        console.print(f"[dim]Eval period: {start_date} to {end_date}[/dim]\n")
    else:
        start_date = (datetime.now() - timedelta(days=eval_days)).strftime('%Y-%m-%d')

    # Progressive training - like YOLO epochs
    console.print(f"[bold magenta]Starting Progressive Training ({args.epochs} epochs)[/bold magenta]\n")

    # Calculate estimators per epoch
    estimators_per_epoch = max(10, args.n_estimators // args.epochs)

    # Warn if configuration is suboptimal
    if args.n_estimators < args.epochs * 10:
        console.print(f"[yellow]Warning: n_estimators ({args.n_estimators}) < epochs ({args.epochs}) * 10[/yellow]")
        console.print(f"[yellow]Consider using --n-estimators {args.epochs * 10} or --epochs {args.n_estimators // 10}[/yellow]\n")

    # Results table
    results_table = Table(show_header=True, header_style="bold cyan")
    results_table.add_column("Epoch", justify="right", style="cyan")
    results_table.add_column("Trees", justify="right")
    results_table.add_column("Win Rate", justify="right")
    results_table.add_column("W/L", justify="right")
    results_table.add_column("Trades", justify="right")
    results_table.add_column("Return", justify="right")
    results_table.add_column("Sharpe", justify="right")
    results_table.add_column("Status", justify="left")

    best_score = 0.0
    best_epoch = 0
    best_model = None

    # Early stopping tracking
    degradation_count = 0
    last_score = 0.0
    min_trades_for_best = 50  # Require at least 50 trades to be considered "best"

    def calculate_composite_score(win_rate, total_return, sharpe, trades):
        """
        Composite score optimizing for: high probability + high profitability
        - win_rate: probability of success
        - total_return: absolute profitability
        - sharpe: risk-adjusted returns (consistency)
        - trades: statistical confidence

        Formula: WR * Return * sqrt(Sharpe) * log(1 + trades/100)
        This rewards models that are profitable, consistent, and statistically significant.
        """
        import math
        if sharpe < 0:
            sharpe = 0.01  # Avoid negative sharpe killing score

        trade_confidence = math.log(1 + trades / 100.0)  # Logarithmic to avoid over-weighting
        sharpe_factor = math.sqrt(max(sharpe, 0.01))  # Square root to moderate impact

        return win_rate * total_return * sharpe_factor * trade_confidence

    # CREATE ONE MLBacktest OBJECT - reuse it to leverage caching
    console.print("[yellow]Creating backtester (data will be cached after first epoch)[/yellow]\n")
    bt = MLBacktest(
        model_type='xgboost',
        retrain=True,
        signal_threshold=config.exit.signal_threshold  # Use config threshold
    )
    bt.sl_atr_mult = config.exit.stop_loss_atr_mult
    bt.tp_atr_mult = config.exit.take_profit_atr_mult

    for epoch in range(1, args.epochs + 1):
        current_estimators = min(estimators_per_epoch * epoch, args.n_estimators)

        # Stop if we've reached max estimators (no point repeating same config)
        if epoch > 1 and current_estimators == args.n_estimators:
            prev_estimators = min(estimators_per_epoch * (epoch - 1), args.n_estimators)
            if prev_estimators == current_estimators:
                console.print(f"\n[yellow]Stopping at epoch {epoch-1}: reached n_estimators={args.n_estimators}[/yellow]")
                break

        console.print(f"\n[bold yellow]Epoch {epoch}/{args.epochs}[/bold yellow] - Training with {current_estimators} trees")

        epoch_start = datetime.now()

        # Update config for this epoch
        config.ml.n_estimators = current_estimators

        # Reuse same bt object - pooled data will be cached after first epoch
        results = bt.run(start_date=start_date, end_date=end_date, train_window=train_window)
        epoch_time = (datetime.now() - epoch_start).total_seconds()

        if results and 'win_rate' in results:
            avg_win = abs(results.get('avg_win', 0))
            avg_loss = abs(results.get('avg_loss', 1))
            wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            win_rate = results.get('win_rate', 0) / 100.0
            total_trades = results.get('total_trades', 0)
            total_return = results.get('total_return', 0)
            sharpe = results.get('sharpe_ratio', 0)

            # Calculate composite score (profit × probability × consistency)
            current_score = calculate_composite_score(win_rate, total_return, sharpe, total_trades)

            # Check if best (require minimum trades to avoid lucky flukes)
            status = ""
            if total_trades >= min_trades_for_best and current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_model = bt.global_xgb
                status = "[bold green]NEW BEST[/bold green]"
                degradation_count = 0  # Reset degradation counter
            elif current_score >= best_score * 0.95:  # Within 5%
                status = "[green]~[/green]"
            else:
                status = "[dim]-[/dim]"

            # Track score degradation for early stopping
            if total_trades >= min_trades_for_best:
                if current_score < last_score:
                    degradation_count += 1
                else:
                    degradation_count = 0
                last_score = current_score

            # Add to table
            results_table.add_row(
                str(epoch),
                str(current_estimators),
                f"{win_rate:.1%}",
                f"{wl_ratio:.2f}x",
                str(total_trades),
                f"{total_return:.1f}%",  # total_return already multiplied by 100 in eval.py
                f"{sharpe:.2f}",
                status
            )

            # Save epoch model
            if bt.global_xgb is not None:
                epoch_model_path = f"{session_dir}/epoch_{epoch}.pkl"
                bt.global_xgb.save(epoch_model_path)
            else:
                epoch_model_path = None

            # Save epoch data
            epoch_data = {
                "epoch": epoch,
                "n_estimators": current_estimators,
                "duration_seconds": epoch_time,
                "model_path": epoch_model_path,
                "metrics": {
                    "win_rate": float(win_rate),
                    "wl_ratio": float(wl_ratio),
                    "total_return": float(total_return),
                    "sharpe_ratio": float(sharpe),
                    "total_trades": int(total_trades),
                    "avg_win": float(avg_win),
                    "avg_loss": float(avg_loss),
                    "composite_score": float(current_score)
                },
                "is_best": bool(current_score == best_score)
            }
            session_data["epochs"].append(epoch_data)

            # Live progress display
            console.print(results_table)

            # Early stopping: if performance degrades for 3 consecutive epochs, stop
            if degradation_count >= 3:
                console.print(f"\n[yellow]Early stopping: Performance degraded for {degradation_count} consecutive epochs[/yellow]")
                console.print(f"[yellow]Overfitting detected - stopping at epoch {epoch}[/yellow]")
                break

        else:
            console.print("[red]Epoch failed[/red]")

    # Training complete
    console.print(f"\n[bold green]Training Complete![/bold green]")
    if best_epoch > 0:
        best_epoch_data = session_data["epochs"][best_epoch - 1]
        best_metrics = best_epoch_data['metrics']
        console.print(f"Best model: Epoch {best_epoch} with {best_epoch_data['n_estimators']} trees")
        console.print(f"  Win Rate: {best_metrics['win_rate']:.1%}")
        console.print(f"  Return: {best_metrics['total_return']:.1f}%")
        console.print(f"  Sharpe: {best_metrics['sharpe_ratio']:.2f}")
        console.print(f"  Trades: {best_metrics['total_trades']}")
        console.print(f"  [dim]Composite Score: {best_metrics['composite_score']:.2f}[/dim]\n")
    else:
        console.print(f"[yellow]No valid best model (all epochs had < {min_trades_for_best} trades)[/yellow]\n")

    # Save best model
    if best_model and best_epoch > 0:
        console.print(f"[bold cyan]Saving Champion Model (Epoch {best_epoch})[/bold cyan]")

        # Save to session folder
        session_best_path = f"{session_dir}/best_model.pkl"
        best_model.save(session_best_path)
        console.print(f"  Session: {session_best_path}")

        # Copy to production location (for morning_signals.py, etc.)
        production_path = "models/global_xgb_champion.pkl"
        best_model.save(production_path)
        console.print(f"  Production: {production_path}")

        # Get best epoch data (already retrieved above)
        best_epoch_data = session_data["epochs"][best_epoch - 1]

        # Save metadata
        metadata = {
            'xgboost': {
                'win_rate': float(best_epoch_data['metrics']['win_rate']),
                'wl_ratio': float(best_epoch_data['metrics']['wl_ratio']),
                'total_trades': int(best_epoch_data['metrics']['total_trades']),
                'total_return': float(best_epoch_data['metrics']['total_return']),
                'sharpe_ratio': float(best_epoch_data['metrics']['sharpe_ratio']),
                'sl_atr_mult': float(config.exit.stop_loss_atr_mult),
                'tp_atr_mult': float(config.exit.take_profit_atr_mult),
                'signal_threshold': float(config.exit.signal_threshold),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_count': feature_count,
                'max_depth': args.max_depth,
                'n_estimators': best_epoch_data['n_estimators'],
                'best_epoch': best_epoch,
                'session_id': session_id
            }
        }

        # Save to session folder
        session_metadata_path = f"{session_dir}/metadata.json"
        with open(session_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Save to production location (for morning_signals.py, etc.)
        production_metadata_path = 'models/champion_metadata.json'
        with open(production_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        console.print(f"  Metadata: {session_metadata_path}")
        console.print(f"  Production Metadata: {production_metadata_path}")

    # Save session
    session_data["end_time"] = datetime.now().isoformat()
    session_data["best_epoch"] = best_epoch

    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=2)

    console.print(f"\n[bold cyan]Session saved to: {session_dir}/[/bold cyan]")
    console.print(f"[dim]  - session.json (training log)[/dim]")
    console.print(f"[dim]  - epoch_1.pkl, epoch_2.pkl, ... (models from each epoch)[/dim]")
    console.print(f"[dim]  - best_model.pkl (champion model)[/dim]")
    console.print(f"[dim]  - metadata.json (performance metrics)[/dim]")

if __name__ == "__main__":
    main()
