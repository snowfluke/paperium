#!/usr/bin/env python3
"""
Auto-Training Optimization Loop
Continuously trains and validates models to achieve performance targets (>80% WR)
"""
import sys
import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.storage import DataStorage
from data.fetcher import DataFetcher
from ml.model import EnsembleModel
from scripts.ml_backtest import MLBacktest
from rich.console import Console

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json
from rich.progress import Progress

TARGET_WIN_RATE = 0.80
MIN_SAVE_THRESHOLD = 0.60    # Save the model if it's at least viable
TARGET_MONTHLY_RETURN = 0.15 # 15% monthly = ~200% annual compounded aggressively
METADATA_PATH = 'models/champion_metadata.json'

class AutoTrainer:
    def __init__(self):
        self.storage = DataStorage(config.data.db_path)
        self.champion_metadata = self._load_metadata()
        # Migration: Ensure all keys exist
        for k in ['xgboost', 'gd_sd']:
            if k not in self.champion_metadata:
                self.champion_metadata[k] = {'win_rate': 0.0, 'date': None}
        
    def _load_metadata(self) -> Dict:
        """Load tracking of current best models."""
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'xgboost': {'win_rate': 0.0, 'date': None},
            'gd_sd': {'win_rate': 0.0, 'date': None}
        }

    def _save_metadata(self):
        """Save tracking of current best models."""
        os.makedirs('models', exist_ok=True)
        with open(METADATA_PATH, 'w') as f:
            json.dump(self.champion_metadata, f, indent=4)

    def run_optimization_loop(self):
        """Run the rigorous Train -> Backtest -> Verify loop (Optimized)"""
        console.print("[bold cyan]Starting Optimized Auto-Training Loop[/bold cyan]")
        console.print(f"Current XGB Champion: [green]{self.champion_metadata.get('xgboost', {'win_rate':0})['win_rate']:.1%}[/green]")
        console.print(f"Current GD/SD Champion: [green]{self.champion_metadata.get('gd_sd', {'win_rate':0})['win_rate']:.1%}[/green]")
        
        # Step 0: Initial Data Loading & Feature Calculation (LOAD ONCE)
        console.print("\n[yellow]⏳ Phase 0: One-time Data Preparation...[/yellow]")
        backtester = MLBacktest()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        all_data = backtester._load_data(start_date, end_date, train_window=252)
        if all_data.empty:
            console.print("[red]No data available[/red]")
            return

        # Pre-calculating features once
        ticker_groups = all_data.groupby('ticker')
        processed_data_list = []
        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Pre-calculating features...", total=all_data['ticker'].nunique())
            for ticker, group in ticker_groups:
                group = group.sort_values('date')
                group = backtester._add_features(group)
                processed_data_list.append(group)
                progress.update(task, advance=1)
        
        featured_data = pd.concat(processed_data_list).sort_values(['date', 'ticker'])
        console.print(f"  ✓ Features calculated for {len(featured_data)} records")

        iteration = 0
        all_results = []
        while True:
            iteration += 1
            console.print(f"\n[bold magenta]Optimization Iteration {iteration}[/bold magenta]")
            
            iter_summary = {}
            for model_type in ['xgboost', 'gd_sd']:
                console.print(f"\n[bold cyan]Evaluating Model Type: {model_type.upper()}[/bold cyan]")
                
                # Step 1: Initialize Backtester
                bt = MLBacktest(model_type=model_type)
                
                # Tuning: Try to hit 80% WR by tightening exits
                if iteration > 1:
                    bt.stop_loss_pct = 0.03 + (iteration * 0.005)
                    bt.take_profit_pct = 0.06 - (iteration * 0.005)
                
                # Step 2: Run Backtest with Pre-loaded Data
                results = bt.run(start_date=start_date, end_date=end_date, train_window=252, pre_loaded_data=featured_data)
                
                if results and 'win_rate' in results:
                    monthly_wrs = [m['win_rate'] / 100.0 for m in results.get('monthly_metrics', [])]
                    effective_wr = (sum(monthly_wrs) / len(monthly_wrs) * 0.7) + (min(monthly_wrs) * 0.3) if monthly_wrs else results['win_rate'] / 100.0
                    
                    ret = results['total_return'] / 100.0
                    console.print(f"  [bold]{model_type.upper()} Results:[/bold] Effective WR: {effective_wr:.1%} | Total Return: {ret:.1%}")
                    iter_summary[model_type] = effective_wr
                    
                    # Champion Comparison Logic
                    current_best_wr = self.champion_metadata[model_type]['win_rate']
                    if model_type == 'xgboost':
                        fname = 'xgb'
                    else:
                        fname = 'sd'
                    save_path = f"models/global_{fname}_champion.pkl"
                    
                    if effective_wr > current_best_wr or not os.path.exists(save_path):
                        improvement = (effective_wr - current_best_wr) if current_best_wr > 0 else effective_wr
                        console.print(f"  [bold green]🚀 Improved Performance! (+{improvement:.1%})[/bold green]")
                        
                        if model_type == 'xgboost':
                            bt.global_xgb.save(save_path)
                        else:
                            bt.global_sd.save(save_path)
                            
                        self.champion_metadata[model_type] = {
                            'win_rate': effective_wr,
                            'total_return': ret,
                            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'iteration': iteration
                        }
                        self._save_metadata()
                    else:
                        console.print(f"  [dim]No improvement over current champion ({current_best_wr:.1%}). Not saving.[/dim]")

                    if effective_wr >= TARGET_WIN_RATE:
                        console.print(f"[bold green]🏆 {model_type.upper()} TARGET REACHED![/bold green]")
                else:
                    console.print(f"[red]Backtest failed for {model_type}.[/red]")
            
            all_results.append(iter_summary)
            if any(wr >= TARGET_WIN_RATE for wr in iter_summary.values()):
                console.print("\n[bold green]One or more models have reached the target performance![/bold green]")
                break

            if iteration >= 5:
                break
            time.sleep(1)

if __name__ == "__main__":
    trainer = AutoTrainer()
    trainer.run_optimization_loop()
