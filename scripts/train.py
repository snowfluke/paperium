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
from ml.features import FeatureEngineer

console = Console()

def main():
    parser = argparse.ArgumentParser(description='Iterative Model Training')
    parser.add_argument('--days', type=str, default='365', help='Evaluation period in calendar days or "max"')
    parser.add_argument('--train-window', type=str, default='max', help='Training window in trading days or "max"')
    parser.add_argument('--target', type=float, default=0.85, help='Target Combined Score (0.0 to 1.0)')
    # Training specs
    parser.add_argument('--max-depth', type=int, default=5, help='XGBoost max tree depth (default: 5)')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of boosting rounds (default: 100)')

    parser.add_argument('--force', action='store_true', help='Replace champion if better.')
    parser.add_argument('--max-iter', type=int, default=10, help='Maximum optimization iterations')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--legacy', action='store_true', help='Use legacy Gen 6 features (46 features)')
    parser.add_argument('--gen9', action='store_true', help='Use GEN9 features (81 features)')

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
            eval_days = 365  # Fixed: 1 year evaluation period
            console.print(f"[dim]Auto-setting eval days to {eval_days} (1 year fixed)[/dim]")
        else:
            eval_days = int(args.days)

        if args.train_window == 'max':
            train_window = int(252 * 4)  # Fixed: 4 years of trading days (1008 days)
            console.print(f"[dim]Auto-setting train window to {train_window} days (4 years fixed)[/dim]")
        else:
            train_window = int(args.train_window)
    else:
        eval_days = int(args.days)
        train_window = int(args.train_window)

    # Determine generation (simple: default to GEN7/8)
    if args.legacy:
        gen_label = "GEN 6"
        use_gen7 = False
        use_gen9 = False
    elif args.gen9:
        gen_label = "GEN 9"
        use_gen7 = True
        use_gen9 = True
    else:
        gen_label = "GEN 7/8"
        use_gen7 = True
        use_gen9 = False

    console.print(f"[bold cyan]Starting Targeted Training for XGBOOST ({gen_label})[/bold cyan]")
    console.print(f"Target Score: [green]{args.target:.1%}[/green] | Max Iter: {args.max_iter}")

    # Initialize training session file
    import json
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_file = f"models/training_session_{gen_label.lower().replace(' ', '_')}_{session_id}.json"
    session_data = {
        "session_id": session_id,
        "generation": gen_label,
        "start_time": datetime.now().isoformat(),
        "parameters": {
            "target_score": args.target,
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "eval_days": eval_days,
            "train_window": train_window,
            "use_gpu": args.gpu,
            "force_replace": args.force,
            "use_gen9_features": use_gen9,
            "use_gen7_features": not args.legacy
        },
        "iterations": []
    }

    # Show feature set banner
    feature_eng = FeatureEngineer(config.ml, use_gen7_features=use_gen7, use_hour0_features='auto', use_gen9_features=use_gen9)
    feature_count = len(feature_eng.feature_set)

    console.print(f"[bold green]✓ {gen_label} ({feature_count} features)[/bold green]")
    if use_gen9:
        console.print("  • Supply/Demand + Microstructure + Order Flow")
    elif use_gen7:
        console.print("  • Intraday Proxies + Crash Filter + Hour-0 (auto-detect)")
    else:
        console.print("  • Basic technical indicators")

    console.print(f"[dim]Session file: {session_file}[/dim]")

    # Show GPU warning once at the start
    if args.gpu:
        import sys
        if sys.platform == "darwin":
            console.print("[yellow]⚠ XGBoost MPS (Metal) acceleration can be unstable on some Mac environments. Using high-performance CPU ('hist') instead.[/yellow]")

    # Phase 0: Data Prep
    backtester = MLBacktest(use_gen7_features=use_gen7, use_gen9_features=use_gen9)
    end_date = datetime.now().strftime('%Y-%m-%d')

    # When using 'max', align to month boundaries for cleaner evaluation
    if args.days == 'max':
        # Start from December 1st of the previous year
        current_year = datetime.now().year
        start_date = f"{current_year - 1}-12-01"
        console.print(f"[dim]Eval period aligned to: {start_date} to {end_date}[/dim]")
    else:
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

    # Gen 6: Improved training with composite scoring and smart optimization
    import numpy as np

    def calculate_composite_score(results, min_trades=52):
        """
        Composite score balancing all objectives:
        Score = Win_Rate * Trade_Penalty * DD_Penalty * WL_Bonus
        
        - Win_Rate: raw win rate (target 0.80+)
        - Trade_Penalty: 1.0 if trades >= min_trades, else trades/min_trades
        - DD_Penalty: 1.0 if DD <= -10%, else penalized
        - WL_Bonus: Bonus for high W/L ratio
        """
        wr = results.get('win_rate', 0) / 100.0
        trades = results.get('total_trades', 0)
        dd = abs(results.get('max_drawdown', 100))
        
        avg_win = abs(results.get('avg_win', 0))
        avg_loss = abs(results.get('avg_loss', 1))
        wl = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Penalties/bonuses
        trade_penalty = min(1.0, trades / min_trades) if min_trades > 0 else 1.0
        dd_penalty = max(0.5, 1.0 - dd / 50) if dd > 10 else 1.0
        wl_bonus = 1.0 + max(0, (wl - 1.0) * 0.1)
        
        return wr * trade_penalty * dd_penalty * wl_bonus

    # Grid search configurations for first pass
    sl_values = [1.5, 2.0, 2.5, 3.0]
    tp_values = [2.0, 3.0, 4.0, 5.0]
    grid_configs = [(sl, tp) for sl in sl_values for tp in tp_values]  # 16 combinations

    # Tracking best configuration
    best_score = 0.0
    best_wr = 0.0
    best_sl_mult = 2.0
    best_tp_mult = 3.0
    best_config = {'sl': best_sl_mult, 'tp': best_tp_mult, 'wr': 0.0, 'wl': 0.0, 'score': 0.0}
    no_improve_count = 0  # For early stopping

    # Create backtester ONCE (preserve cache across iterations)
    # Use moderate threshold during training for balanced feedback
    bt = MLBacktest(model_type='xgboost', retrain=True, signal_threshold=0.40)

    iteration = 0
    while iteration < args.max_iter:
        iteration += 1

        # Phase 1 (1-16): Grid search through predefined configurations
        # Phase 2 (17+): Fine-tune around best found configuration
        if iteration <= len(grid_configs):
            sl_mult, tp_mult = grid_configs[iteration - 1]
        else:
            # Fine-tune around best with small perturbations
            sl_mult = float(np.clip(best_sl_mult + np.random.uniform(-0.2, 0.2), 1.5, 3.5))
            tp_mult = float(np.clip(best_tp_mult + np.random.uniform(-0.3, 0.3), 2.0, 5.0))

        console.print(f"\n[bold magenta]Iteration {iteration}/{args.max_iter}[/bold magenta]")
        phase = "Grid Search" if iteration <= len(grid_configs) else "Fine-Tuning"
        console.print(f"  Phase: [cyan]{phase}[/cyan] | SL/TP: [cyan]{sl_mult:.2f}x / {tp_mult:.2f}x ATR[/cyan]")

        # Update SL/TP for this iteration
        bt.sl_atr_mult = sl_mult
        bt.tp_atr_mult = tp_mult

        iteration_start = datetime.now()
        results = bt.run(start_date=start_date, end_date=end_date, train_window=train_window, pre_loaded_data=featured_data)
        iteration_time = (datetime.now() - iteration_start).total_seconds()

        if results and 'win_rate' in results:
            # Calculate metrics
            avg_win = abs(results.get('avg_win', 0))
            avg_loss = abs(results.get('avg_loss', 1))
            wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            effective_wr = results.get('win_rate', 0) / 100.0
            total_trades = results.get('total_trades', 0)
            max_dd = abs(results.get('max_drawdown', 0))
            
            # Calculate composite score
            composite_score = calculate_composite_score(results)
            
            console.print(f"  Win Rate: [bold]{effective_wr:.1%}[/bold] | Trades: {total_trades} | W/L: {wl_ratio:.2f}x | DD: {max_dd:.1f}%")
            console.print(f"  [yellow]Composite Score: {composite_score:.4f}[/yellow]")

            # Track if this is the best so far
            if composite_score > best_score * 1.001:  # 0.1% improvement threshold
                best_score = composite_score
                best_wr = effective_wr
                best_sl_mult = sl_mult
                best_tp_mult = tp_mult
                best_config = {
                    'sl': sl_mult, 'tp': tp_mult, 
                    'wr': effective_wr, 'wl': wl_ratio, 
                    'score': composite_score, 'trades': total_trades
                }
                no_improve_count = 0
                console.print(f"  [green]★ New Best! Score: {composite_score:.4f}[/green]")
            else:
                no_improve_count += 1
                console.print(f"  [dim]No improvement ({no_improve_count}/5)[/dim]")

            # Save iteration data to session file
            iteration_data = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": iteration_time,
                "phase": phase,
                "sl_tp_config": {
                    "sl_atr_mult": float(sl_mult),
                    "tp_atr_mult": float(tp_mult)
                },
                "metrics": {
                    "win_rate": float(effective_wr),
                    "wl_ratio": float(wl_ratio),
                    "composite_score": float(composite_score),
                    "total_return": float(results.get('total_return', 0)),
                    "sharpe_ratio": float(results.get('sharpe_ratio', 0)),
                    "sortino_ratio": float(results.get('sortino_ratio', 0)),
                    "max_drawdown": float(results.get('max_drawdown', 0)),
                    "total_trades": int(total_trades),
                    "avg_win": float(avg_win),
                    "avg_loss": float(avg_loss)
                },
                "monthly_performance": results.get('monthly_metrics', []),
                "exit_breakdown": results.get('exit_breakdown', {})
            }
            session_data["iterations"].append(iteration_data)

            # Save session file after each iteration
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)

            # Save champion if this is the best and --force is set
            metadata_path = 'models/champion_metadata.json'
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    metadata = {'xgboost': {'composite_score': 0.0}}
            else:
                metadata = {'xgboost': {'composite_score': 0.0}}
                
            current_best_score = metadata.get('xgboost', {}).get('composite_score', 0.0)

            if args.force and composite_score > current_best_score:
                console.print(f"  [green]Champion Updated! (Score: {current_best_score:.4f} → {composite_score:.4f})[/green]")
                save_path = "models/global_xgb_champion.pkl"
                bt.global_xgb.save(save_path)

                metadata['xgboost'] = {
                    'win_rate': float(effective_wr),
                    'wl_ratio': float(wl_ratio),
                    'composite_score': float(composite_score),
                    'total_trades': int(total_trades),
                    'sl_atr_mult': float(sl_mult),
                    'tp_atr_mult': float(tp_mult),
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'target_met': bool(effective_wr >= args.target)
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            # Check if target reached
            if effective_wr >= args.target and total_trades >= 52:
                console.print(f"\n[bold green]🎯 Target reached! WR={effective_wr:.1%}, Trades={total_trades}[/bold green]")
                break
            
            # Early stopping: no improvement for 5 consecutive iterations (after grid search)
            if iteration > len(grid_configs) and no_improve_count >= 5:
                console.print(f"\n[yellow]⚠ Early stopping: No improvement for 5 iterations[/yellow]")
                break
        else:
            console.print("[red]Backtest iteration failed.[/red]")

    # Save final session summary
    session_data["end_time"] = datetime.now().isoformat()
    session_data["total_iterations"] = iteration

    if session_data["iterations"]:
        best_iteration = max(session_data["iterations"], key=lambda x: x["metrics"]["win_rate"])
        session_data["best_iteration"] = {
            "iteration": best_iteration["iteration"],
            "win_rate": best_iteration["metrics"]["win_rate"],
            "wl_ratio": best_iteration["metrics"]["wl_ratio"],
            "total_return": best_iteration["metrics"]["total_return"],
            "sl_atr_mult": best_iteration["sl_tp_config"]["sl_atr_mult"],
            "tp_atr_mult": best_iteration["sl_tp_config"]["tp_atr_mult"]
        }

    # Gen 5.1: Save best SL/TP configuration
    session_data["best_sl_tp_config"] = best_config

    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=2)

    console.print(f"\n[bold cyan]Training session completed![/bold cyan]")
    console.print(f"Session data saved to: [dim]{session_file}[/dim]")

if __name__ == "__main__":
    main()
