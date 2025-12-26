#!/usr/bin/env python3
"""
Targeted Model Training CLI
Allows training a specific model (XGB or GD/SD) with a custom performance target.
"""
import sys
import os
import argparse
import time
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from scripts.ml_backtest import MLBacktest
from scripts.auto_train import AutoTrainer
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(description='Targeted Model Training')
    parser.add_argument('--type', choices=['xgboost', 'gd_sd'], required=True, help='Model type to train')
    parser.add_argument('--target', type=float, default=0.80, help='Target effective Win Rate (0.0 to 1.0)')
    parser.add_argument('--force', action='store_true', help='Ignore existing champion metrics')
    parser.add_argument('--max-iter', type=int, default=5, help='Maximum optimization iterations')

    args = parser.parse_args()
    
    trainer = AutoTrainer()
    if args.force:
        console.print(f"[yellow]Force mode enabled. Ignoring existing {args.type} champion metrics.[/yellow]")
        trainer.champion_metadata[args.type]['win_rate'] = 0.0

    console.print(f"[bold cyan]Starting Targeted Training for {args.type.upper()}[/bold cyan]")
    console.print(f"Target Win Rate: [green]{args.target:.1%}[/green]")

    # Phase 0: Data Prep
    backtester = MLBacktest()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    all_data = backtester._load_data(start_date, end_date, train_window=252)
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
        
        bt = MLBacktest(model_type=args.type)
        if iteration > 1:
            bt.stop_loss_pct = 0.03 + (iteration * 0.005)
            bt.take_profit_pct = 0.06 - (iteration * 0.005)
            
        results = bt.run(start_date=start_date, end_date=end_date, train_window=252, pre_loaded_data=featured_data)
        
        if results and 'win_rate' in results:
            monthly_wrs = [m['win_rate'] / 100.0 for m in results.get('monthly_metrics', [])]
            effective_wr = (sum(monthly_wrs) / len(monthly_wrs) * 0.7) + (min(monthly_wrs) * 0.3) if monthly_wrs else results['win_rate'] / 100.0
            
            console.print(f"  Effective WR: [bold]{effective_wr:.1%}[/bold]")
            
            # Comparison
            current_best_wr = trainer.champion_metadata[args.type]['win_rate']
            if args.type == 'xgboost':
                fname = 'xgb'
            else:
                fname = 'sd'
            save_path = f"models/global_{fname}_champion.pkl"
            
            if effective_wr > current_best_wr:
                console.print(f"  [green]🚀 Champion Updated! ({current_best_wr:.1%} -> {effective_wr:.1%})[/green]")
                if args.type == 'xgboost':
                    bt.global_xgb.save(save_path)
                else:
                    bt.global_sd.save(save_path)
                    
                trainer.champion_metadata[args.type] = {
                    'win_rate': effective_wr,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'target_met': effective_wr >= args.target
                }
                trainer._save_metadata()
            
            if effective_wr >= args.target:
                console.print(f"[bold green]Target reached! Optimization complete.[/bold green]")
                break
        else:
            console.print("[red]Backtest iteration failed.[/red]")

if __name__ == "__main__":
    main()
