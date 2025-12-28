#!/usr/bin/env python3
"""
Plot Training Session Progression
Visualizes metrics from training session files.
"""
import json
import sys
import argparse
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rich.console import Console

console = Console()

def load_session(session_file: str) -> dict:
    """Load training session data from JSON file."""
    with open(session_file, 'r') as f:
        return json.load(f)

def plot_session(session_data: dict, save_path: Optional[str] = None):
    """Create comprehensive training progression plots."""
    iterations = session_data.get('iterations', [])

    if not iterations:
        console.print("[red]No iteration data found in session file.[/red]")
        return

    # Extract metrics
    iter_nums = [it['iteration'] for it in iterations]
    win_rates = [it['metrics']['win_rate'] * 100 for it in iterations]
    wl_ratios = [it['metrics']['wl_ratio'] for it in iterations]
    total_returns = [it['metrics']['total_return'] * 100 for it in iterations]
    sharpe_ratios = [it['metrics']['sharpe_ratio'] for it in iterations]
    max_drawdowns = [it['metrics']['max_drawdown'] * 100 for it in iterations]
    total_trades = [it['metrics']['total_trades'] for it in iterations]

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Session info
    session_id = session_data.get('session_id', 'Unknown')
    params = session_data.get('parameters', {})
    fig.suptitle(f'Training Session: {session_id}\n'
                 f'Target: {params.get("target_score", 0):.1%} | '
                 f'Max Depth: {params.get("max_depth", "?")} | '
                 f'Estimators: {params.get("n_estimators", "?")}',
                 fontsize=14, fontweight='bold')

    # 1. Win Rate over iterations
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iter_nums, win_rates, marker='o', linewidth=2, markersize=6, color='#2ecc71')
    ax1.axhline(y=params.get('target_score', 0.85) * 100, color='r', linestyle='--',
                label=f'Target: {params.get("target_score", 0.85):.1%}', alpha=0.7)
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Win Rate (%)', fontweight='bold')
    ax1.set_title('Win Rate Progression', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. W/L Ratio over iterations
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iter_nums, wl_ratios, marker='s', linewidth=2, markersize=6, color='#3498db')
    ax2.set_xlabel('Iteration', fontweight='bold')
    ax2.set_ylabel('W/L Ratio', fontweight='bold')
    ax2.set_title('Win/Loss Ratio Progression', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Total Return over iterations
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(iter_nums, total_returns, marker='^', linewidth=2, markersize=6, color='#9b59b6')
    ax3.set_xlabel('Iteration', fontweight='bold')
    ax3.set_ylabel('Total Return (%)', fontweight='bold')
    ax3.set_title('Total Return Progression', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Sharpe Ratio over iterations
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(iter_nums, sharpe_ratios, marker='D', linewidth=2, markersize=6, color='#e74c3c')
    ax4.set_xlabel('Iteration', fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio', fontweight='bold')
    ax4.set_title('Sharpe Ratio Progression', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Max Drawdown over iterations
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(iter_nums, max_drawdowns, marker='v', linewidth=2, markersize=6, color='#f39c12')
    ax5.set_xlabel('Iteration', fontweight='bold')
    ax5.set_ylabel('Max Drawdown (%)', fontweight='bold')
    ax5.set_title('Max Drawdown Progression', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.invert_yaxis()  # Invert so worse drawdowns are lower

    # 6. Total Trades over iterations
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(iter_nums, total_trades, marker='o', linewidth=2, markersize=6, color='#1abc9c')
    ax6.set_xlabel('Iteration', fontweight='bold')
    ax6.set_ylabel('Total Trades', fontweight='bold')
    ax6.set_title('Total Trades per Iteration', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        console.print(f"[green]Plot saved to: {save_path}[/green]")
    else:
        plt.show()

def list_sessions():
    """List all available training session files."""
    models_dir = Path('models')
    session_files = sorted(models_dir.glob('training_session_*.json'), reverse=True)

    if not session_files:
        console.print("[yellow]No training session files found.[/yellow]")
        return []

    console.print("\n[bold cyan]Available Training Sessions:[/bold cyan]\n")
    for i, session_file in enumerate(session_files, 1):
        with open(session_file, 'r') as f:
            data = json.load(f)

        params = data.get('parameters', {})
        iterations = len(data.get('iterations', []))
        best = data.get('best_iteration', {})

        console.print(f"{i}. {session_file.name}")
        console.print(f"   Started: {data.get('start_time', 'Unknown')}")
        console.print(f"   Iterations: {iterations}/{params.get('max_iter', '?')}")
        if best:
            console.print(f"   Best: WR={best.get('win_rate', 0):.1%}, W/L={best.get('wl_ratio', 0):.2f}x")
        console.print()

    return session_files

def main():
    parser = argparse.ArgumentParser(description='Plot training session progression')
    parser.add_argument('session_file', nargs='?', help='Path to training session JSON file')
    parser.add_argument('--save', type=str, help='Save plot to file instead of displaying')
    parser.add_argument('--list', action='store_true', help='List all available session files')

    args = parser.parse_args()

    if args.list or not args.session_file:
        session_files = list_sessions()

        if not args.session_file and session_files:
            # Interactive selection
            try:
                choice = int(input("Select session number to plot (or 0 to exit): "))
                if choice > 0 and choice <= len(session_files):
                    args.session_file = str(session_files[choice - 1])
                else:
                    console.print("[dim]Exiting...[/dim]")
                    return
            except (ValueError, KeyboardInterrupt):
                console.print("\n[dim]Exiting...[/dim]")
                return

    if args.session_file:
        try:
            session_data = load_session(args.session_file)
            plot_session(session_data, save_path=args.save)
        except FileNotFoundError:
            console.print(f"[red]Error: File not found: {args.session_file}[/red]")
        except json.JSONDecodeError:
            console.print(f"[red]Error: Invalid JSON file: {args.session_file}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()
