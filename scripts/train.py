#!/usr/bin/env python3
"""
Targeted Model Training CLI (Gen 5)
Allows training the XGBoost model with a combined performance target (Win Rate + W/L Ratio).
"""
import sys
import os
import argparse
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from rich.console import Console

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from scripts.eval import MLBacktest

console = Console()

def main():
    parser = argparse.ArgumentParser(description='Targeted Model Training (Gen 5)')
    parser.add_argument('--days', type=str, default='365', help='Evaluation period in calendar days or "max"')
    parser.add_argument('--train-window', type=str, default='max', help='Training window in trading days or "max"')
    parser.add_argument('--target', type=float, default=0.85, help='Target Combined Score (0.0 to 1.0)')
    # Gen 5 Ultimate Specs
    parser.add_argument('--max-depth', type=int, default=5, help='XGBoost max tree depth (Conservative: 5)')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of boosting rounds (Conservative: 100)')

    parser.add_argument('--force', action='store_true', help='Replace champion if better.')
    parser.add_argument('--max-iter', type=int, default=10, help='Maximum optimization iterations')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')

    args = parser.parse_args()
    
    # Update config with CLI args
    config.ml.use_gpu = args.gpu
    config.ml.max_depth = args.max_depth
    config.ml.n_estimators = args.n_estimators
    
    # Process 'max' arguments

    if args.days == 'max' or args.train_window == 'max':
        conn = sqlite3.connect(config.data.db_path)
        df_dates = pd.read_sql("SELECT MIN(date), MAX(date) FROM prices", conn)
        conn.close()
        
        db_min = pd.to_datetime(str(df_dates.iloc[0, 0]))
        db_max = pd.to_datetime(str(df_dates.iloc[0, 1]))
        total_days = (db_max - db_min).days
        
        if args.days == 'max':
            if total_days >= 365 * 1.5:
                eval_days = 365
            else:
                eval_days = max(30, total_days // 3)
            console.print(f"[dim]Auto-setting eval days to {eval_days} (max window)[/dim]")
        else:
            eval_days = int(args.days)
            
        if args.train_window == 'max':
            train_window = min(total_days - eval_days, 252 * 5)
            console.print(f"[dim]Auto-setting train window to {train_window} (max window)[/dim]")
        else:
            train_window = int(args.train_window)
    else:
        eval_days = int(args.days)
        train_window = int(args.train_window)

    console.print(f"[bold cyan]Starting Targeted Training for XGBOOST (Gen 5)[/bold cyan]")
    console.print(f"Target Score: [green]{args.target:.1%}[/green] | Max Iter: {args.max_iter}")

    # Phase 0: Data Prep
    backtester = MLBacktest()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=eval_days)).strftime('%Y-%m-%d')
    
    all_data = backtester._load_data(start_date, end_date, train_window=train_window)
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
    console.print(f"  ✓ Loaded {len(featured_data)} records for {all_data['ticker'].nunique()} stocks")

    iteration = 0
    while iteration < args.max_iter:
        iteration += 1
        console.print(f"\n[bold magenta]Iteration {iteration}/{args.max_iter}[/bold magenta]")
        
        bt = MLBacktest(model_type='xgboost', retrain=True)
        if iteration > 1:
            # Exact Gen 4 Iteration Logic
            bt.stop_loss_pct = 0.03 + (iteration * 0.005)
            bt.take_profit_pct = 0.06 - (iteration * 0.005)
            console.print(f"  [dim]Optimization tuning: SL={bt.stop_loss_pct:.1%}, TP={bt.take_profit_pct:.1%}[/dim]")
            
        results = bt.run(start_date=start_date, end_date=end_date, train_window=train_window, pre_loaded_data=featured_data)
        
        if results and 'win_rate' in results:
            # Calculate Gen 4/5 Metrics
            monthly_wrs = [m['win_rate'] / 100.0 for m in results.get('monthly_metrics', [])]
            effective_wr = (sum(monthly_wrs) / len(monthly_wrs) * 0.7) + (min(monthly_wrs) * 0.3) if monthly_wrs else results['win_rate'] / 100.0
            
            avg_win = abs(results.get('avg_win', 0))
            avg_loss = abs(results.get('avg_loss', 1))
            wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Gen 5 Combined Score: prioritize quality
            wl_ratio_normalized = min(wl_ratio / 2.5, 1.0) 
            combined_score = (effective_wr * 0.4) + (wl_ratio_normalized * 0.6)
            
            console.print(f"  Win Rate: [bold]{effective_wr:.1%}[/bold]")
            console.print(f"  W/L Ratio: [bold]{wl_ratio:.2f}x[/bold]")
            console.print(f"  [cyan]Combined Score: {combined_score:.1%}[/cyan]")
            
            # Metadata for comparison
            import json
            metadata_path = 'models/champion_metadata.json'
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    metadata = {'xgboost': {'combined_score': 0.0}}
            else:
                metadata = {'xgboost': {'combined_score': 0.0}}
                
            # Compare using Gen 5 score
            current_best_score = metadata.get('xgboost', {}).get('combined_score', 0.0)
            
            if args.force:
                if combined_score > current_best_score:
                    console.print(f"  [green]Champion Updated! ({current_best_score:.1%} -> {combined_score:.1%})[/green]")
                    save_path = "models/global_xgb_champion.pkl"
                    bt.global_xgb.save(save_path)
                    
                    metadata['xgboost'] = {
                        'win_rate': float(effective_wr),
                        'wl_ratio': float(wl_ratio),
                        'combined_score': float(combined_score),
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'target_met': bool(combined_score >= args.target),
                        'hyperparams': {
                            'stop_loss': float(bt.stop_loss_pct),
                            'take_profit': float(bt.take_profit_pct)
                        }
                    }
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                else:
                    console.print(f"  [yellow]No improvement over champ ({current_best_score:.1%}).[/yellow]")
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
