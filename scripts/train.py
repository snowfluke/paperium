#!/usr/bin/env python3
"""
Targeted Model Training CLI
Allows training the XGBoost model with a custom performance target.
"""
import sys
import os
import argparse
import time
from datetime import datetime, timedelta
import pandas as pd
import sqlite3 # Added for 'max' argument processing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from scripts.eval import MLBacktest # Changed from ml_backtest
# Removed: from scripts.auto_train import AutoTrainer
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description='Targeted Model Training')
    parser.add_argument('--days', type=str, default='90', help='Evaluation period in calendar days or "max"')
    parser.add_argument('--train-window', type=str, default='252', help='Training window in trading days or "max"')
    parser.add_argument('--target', type=float, default=0.85, help='Target combined score (Win Rate + W/L Ratio, 0.0 to 1.0)')
    parser.add_argument('--force', action='store_true', help='Replace champion if better. If False, saves with a new name.')
    parser.add_argument('--max-iter', type=int, default=5, help='Maximum optimization iterations')
    parser.add_argument('--max-depth', type=int, default=5, help='XGBoost max tree depth')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of boosting rounds')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='XGBoost learning rate')

    args = parser.parse_args()
    
    # Process 'max' arguments
    if args.days == 'max' or args.train_window == 'max':
        conn = sqlite3.connect(config.data.db_path)
        df_dates = pd.read_sql("SELECT MIN(date), MAX(date) FROM prices", conn)
        conn.close()
        
        db_min = pd.to_datetime(df_dates.iloc[0, 0])
        db_max = pd.to_datetime(df_dates.iloc[0, 1])
        total_days = (db_max - db_min).days
        
        if args.days == 'max':
            eval_days = min(total_days // 4, 365) # Max 1 year for eval
            console.print(f"[dim]Auto-setting eval days to {eval_days} (max window)[/dim]")
        else:
            eval_days = int(args.days)
            
        if args.train_window == 'max':
            train_window = min(total_days, 252 * 3) # Max 3 years for training
            console.print(f"[dim]Auto-setting train window to {train_window} (max window)[/dim]")
        else:
            train_window = int(args.train_window)
    else:
        eval_days = int(args.days)
        train_window = int(args.train_window)

    # Removed: trainer = AutoTrainer()
    # Removed: if args.force:
    # Removed:     console.print(f"[yellow]Force mode enabled. Ignoring existing {args.type} champion metrics.[/yellow]")
    # Removed:     trainer.champion_metadata[args.type]['win_rate'] = 0.0

    console.print(f"[bold cyan]Starting Targeted Training for XGBOOST[/bold cyan]") # Changed from args.type.upper()
    console.print(f"Target Win Rate: [green]{args.target:.1%}[/green]")

    # Phase 0: Data Prep
    backtester = MLBacktest()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=eval_days)).strftime('%Y-%m-%d') # Using eval_days
    
    all_data = backtester._load_data(start_date, end_date, train_window=train_window) # Using train_window
    if all_data.empty:
        console.print("[red]No data available[/red]")
        return

    # Pre-calculating features
    ticker_groups = all_data.groupby('ticker')
    processed_data_list = []
    for ticker, group in ticker_groups:
        group = group.sort_values('date')
        group = backtester._add_features(group)
        processed_data_list.append(group)
    featured_data = pd.concat(processed_data_list).sort_values(['date', 'ticker'])

    iteration = 0
    while iteration < args.max_iter:
        iteration += 1
        console.print(f"\n[bold magenta]Iteration {iteration}/{args.max_iter}[/bold magenta]")
        
        bt = MLBacktest(model_type='xgboost', retrain=True)
        if iteration > 1:
            # Optimized for better Win/Loss ratio: tighter stops, wider targets
            bt.stop_loss_pct = 0.025 + (iteration * 0.003)   # 2.5% → 4.0%
            bt.take_profit_pct = 0.10 - (iteration * 0.008)  # 10% → 6.4%
            
        results = bt.run(start_date=start_date, end_date=end_date, train_window=train_window, pre_loaded_data=featured_data)
        
        if results and 'win_rate' in results:
            monthly_wrs = [m['win_rate'] / 100.0 for m in results.get('monthly_metrics', [])]
            effective_wr = (sum(monthly_wrs) / len(monthly_wrs) * 0.7) + (min(monthly_wrs) * 0.3) if monthly_wrs else results['win_rate'] / 100.0
            
            # Calculate Win/Loss ratio for optimization
            avg_win = abs(results.get('avg_win', 0))
            avg_loss = abs(results.get('avg_loss', 1))
            wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Combined score: prioritize W/L ratio (60%) + win rate (40%)
            wl_ratio_normalized = min(wl_ratio / 2.5, 1.0)  # Target 2.5x W/L ratio
            combined_score = (effective_wr * 0.4) + (wl_ratio_normalized * 0.6)
            
            console.print(f"  Win Rate: [bold]{effective_wr:.1%}[/bold]")
            console.print(f"  Win/Loss Ratio: [bold]{wl_ratio:.2f}x[/bold]")
            console.print(f"  Combined Score: [bold cyan]{combined_score:.1%}[/bold cyan]")
            
            # Metadata for comparison
            import json
            metadata_path = 'models/champion_metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {'xgboost': {'win_rate': 0.0}}
                
            current_best_wr = metadata.get('xgboost', {}).get('win_rate', 0.0)
            
            if args.force:
                # Compare using combined score instead of just win rate
                current_best_score = metadata.get('xgboost', {}).get('combined_score', current_best_wr)
                
                if combined_score > current_best_score:
                    console.print(f"  [green]Champion Updated! Score: {current_best_score:.1%} -> {combined_score:.1%}[/green]")
                    save_path = "models/global_xgb_champion.pkl"
                    bt.global_xgb.save(save_path)
                    
                    metadata['xgboost'] = {
                        'win_rate': effective_wr,
                        'wl_ratio': wl_ratio,
                        'combined_score': combined_score,
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'target_met': combined_score >= args.target,
                        'hyperparams': {
                            'max_depth': args.max_depth,
                            'n_estimators': args.n_estimators,
                            'learning_rate': args.learning_rate
                        }
                    }
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                else:
                    console.print(f"  [yellow]No improvement over current champion (score: {current_best_score:.1%}). Not replacing.[/yellow]")
            else:
                # Save with new name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                save_path = f"models/xgb_model_{timestamp}.pkl"
                bt.global_xgb.save(save_path)
                console.print(f"  [blue]Model saved to {save_path}[/blue]")
            
            if combined_score >= args.target:
                console.print(f"[bold green]Target reached! Optimization complete.[/bold green]")
                console.print(f"  Final: WR={effective_wr:.1%}, W/L={wl_ratio:.2f}x, Score={combined_score:.1%}")
                break
        else:
            console.print("[red]Backtest iteration failed.[/red]")

if __name__ == "__main__":
    main()
