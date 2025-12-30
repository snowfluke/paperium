#!/usr/bin/env python3
"""
Progressive Training System (Like YOLO/RL)
Iteratively adds features and keeps only what improves performance.

Philosophy: Start simple, add complexity only when it helps.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.progress import track

from config import config
from scripts.eval import MLBacktest

console = Console()


class ProgressiveTrainer:
    """
    Trains model progressively, testing feature groups one at a time.
    Only keeps features that improve win rate.
    """

    def __init__(self, target_wr=0.90):
        self.target_wr = target_wr
        self.best_wr = 0.0
        self.best_config = {}
        self.history = []

        # Feature groups to test (in order)
        self.feature_groups = {
            'base': {
                'use_gen7': False,
                'use_gen9': False,
                'label': 'BASE (46 features)'
            },
            'gen7': {
                'use_gen7': True,
                'use_gen9': False,
                'label': 'GEN7 (51 features)'
            },
            'gen8': {
                'use_gen7': True,
                'use_gen9': False,
                'label': 'GEN8 (56 features, auto-detect Hour-0)'
            },
            'gen9': {
                'use_gen7': True,
                'use_gen9': True,
                'label': 'GEN9 (81 features)'
            }
        }

        # SL/TP to test
        self.sl_tp_configs = [
            (1.5, 3.0),
            (2.0, 4.0),
            (2.5, 5.0),
        ]

    def test_configuration(self, feature_config, sl_mult, tp_mult, start_date, end_date, train_window):
        """Test a single configuration."""
        try:
            bt = MLBacktest(
                retrain=True,
                use_gen7_features=feature_config['use_gen7'],
                use_gen9_features=feature_config['use_gen9'],
                sl_atr_mult=sl_mult,
                tp_atr_mult=tp_mult,
                signal_threshold=0.50  # Moderate threshold
            )

            results = bt.run(
                start_date=start_date,
                end_date=end_date,
                train_window=train_window
            )

            if results and 'win_rate' in results:
                wr = results['win_rate'] / 100.0
                trades = results['total_trades']
                wl = results.get('avg_win', 0) / abs(results.get('avg_loss', 1))
                return_pct = results.get('total_return', 0)

                return {
                    'win_rate': wr,
                    'trades': trades,
                    'wl_ratio': wl,
                    'return': return_pct,
                    'success': True
                }
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        return {'success': False}

    def run(self, days=365, train_window=1008):
        """Run progressive training."""
        console.print("[bold cyan]Progressive Training System[/bold cyan]")
        console.print("Testing feature groups iteratively...\n")

        # Setup dates
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        best_overall_wr = 0.0
        best_overall_config = None

        # Test each feature group
        for group_name, feature_config in self.feature_groups.items():
            console.print(f"\n[bold yellow]Testing: {feature_config['label']}[/bold yellow]")

            best_group_wr = 0.0
            best_group_result = None

            # Test SL/TP combinations for this feature group
            for sl, tp in track(self.sl_tp_configs, description="Optimizing SL/TP..."):
                result = self.test_configuration(
                    feature_config, sl, tp,
                    start_date, end_date, train_window
                )

                if result['success']:
                    wr = result['win_rate']
                    console.print(
                        f"  SL/TP {sl:.1f}x/{tp:.1f}x: "
                        f"WR={wr:.1%}, Trades={result['trades']}, "
                        f"W/L={result['wl_ratio']:.2f}x, Return={result['return']:.1f}%"
                    )

                    if wr > best_group_wr:
                        best_group_wr = wr
                        best_group_result = {
                            'group': group_name,
                            'label': feature_config['label'],
                            'sl': sl,
                            'tp': tp,
                            **result
                        }

            # Compare with overall best
            if best_group_result and best_group_wr > best_overall_wr:
                improvement = (best_group_wr - best_overall_wr) * 100
                console.print(
                    f"\n[bold green]✓ Improvement: {improvement:.1f}% "
                    f"({best_overall_wr:.1%} → {best_group_wr:.1%})[/bold green]"
                )
                best_overall_wr = best_group_wr
                best_overall_config = best_group_result
                self.history.append(best_group_result)
            else:
                console.print(
                    f"\n[dim]No improvement from {feature_config['label']}, "
                    f"keeping previous best[/dim]"
                )
                # Early stop if adding features hurts performance
                if best_group_wr < best_overall_wr * 0.95:
                    console.print("[yellow]Performance degraded >5%, stopping feature additions[/yellow]")
                    break

        # Summary
        console.print("\n" + "="*60)
        console.print("[bold cyan]Progressive Training Complete![/bold cyan]\n")

        if best_overall_config:
            console.print(f"[bold green]Best Configuration:[/bold green]")
            console.print(f"  Features: {best_overall_config['label']}")
            console.print(f"  Win Rate: {best_overall_config['win_rate']:.1%}")
            console.print(f"  W/L Ratio: {best_overall_config['wl_ratio']:.2f}x")
            console.print(f"  Total Return: {best_overall_config['return']:.1f}%")
            console.print(f"  SL/TP: {best_overall_config['sl']:.1f}x / {best_overall_config['tp']:.1f}x ATR")
            console.print(f"  Total Trades: {best_overall_config['trades']}")

            if best_overall_config['win_rate'] >= self.target_wr:
                console.print(f"\n[bold green]🎯 TARGET {self.target_wr:.0%} REACHED![/bold green]")
            else:
                gap = (self.target_wr - best_overall_config['win_rate']) * 100
                console.print(f"\n[yellow]Gap to target: {gap:.1f}%[/yellow]")

        # Show progression
        if len(self.history) > 1:
            console.print("\n[bold cyan]Improvement History:[/bold cyan]")
            for i, h in enumerate(self.history, 1):
                console.print(
                    f"  {i}. {h['label']}: {h['win_rate']:.1%} "
                    f"(SL/TP: {h['sl']:.1f}x/{h['tp']:.1f}x)"
                )

        return best_overall_config


def main():
    parser = argparse.ArgumentParser(description='Progressive Training (Iterative Improvement)')
    parser.add_argument('--days', type=int, default=365, help='Evaluation period')
    parser.add_argument('--train-window', type=int, default=1008, help='Training window (4 years)')
    parser.add_argument('--target', type=float, default=0.90, help='Target win rate')

    args = parser.parse_args()

    trainer = ProgressiveTrainer(target_wr=args.target)
    best = trainer.run(days=args.days, train_window=args.train_window)

    if best:
        console.print(f"\n[dim]Use this config in your next training:[/dim]")
        use_gen9 = "--gen9" if best['group'] == 'gen9' else "--no-gen9"
        console.print(
            f"  uv run python scripts/train.py {use_gen9} "
            f"--target {best['win_rate']:.2f} --force"
        )


if __name__ == "__main__":
    main()
