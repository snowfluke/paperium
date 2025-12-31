#!/usr/bin/env python3
"""
Terminal-based Training Session Viewer
Displays training session progression in the terminal using Rich.
"""
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def create_sparkline(values: list, width: int = 30, min_val: Optional[float] = None, max_val: Optional[float] = None) -> str:
    """Create a simple ASCII sparkline chart."""
    if not values:
        return ""

    # Sparkline characters from lowest to highest
    chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

    min_v = min_val if min_val is not None else min(values)
    max_v = max_val if max_val is not None else max(values)

    if max_v == min_v:
        return chars[4] * min(len(values), width)

    # Normalize and map to characters
    sparkline = ""

    for val in values[-width:]:  # Show last 'width' values
        normalized = (val - min_v) / (max_v - min_v) if max_v != min_v else 0.5
        idx = min(int(normalized * (len(chars) - 1)), len(chars) - 1)
        sparkline += chars[idx]

    return sparkline

def display_session(session_file: str):
    """Display training session data in terminal."""
    try:
        with open(session_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading session file: {e}[/red]")
        return

    # Handle both old (iterations) and new (epochs) structure
    epochs = data.get('epochs', [])
    iterations = data.get('iterations', [])
    items = epochs if epochs else iterations

    if not items:
        console.print("[yellow]No training data found.[/yellow]")
        return

    # Extract metrics (compatible with both structures)
    win_rates = [it['metrics']['win_rate'] for it in items]
    wl_ratios = [it['metrics']['wl_ratio'] for it in items]
    total_returns = [it['metrics']['total_return'] for it in items]
    sharpe_ratios = [it['metrics']['sharpe_ratio'] for it in items]
    max_drawdowns = [it['metrics'].get('max_drawdown', 0) for it in items]  # May not exist in epoch structure

    # Session header
    params = data.get('parameters', {})
    session_id = data.get('session_id', 'Unknown')
    item_type = "Epochs" if epochs else "Iterations"

    header = Panel(
        f"[bold cyan]Training Session: {session_id}[/bold cyan]\n"
        f"[dim]Started: {data.get('start_time', 'Unknown')}[/dim]\n"
        f"[dim]Max Depth: {params.get('max_depth', '?')} | "
        f"Estimators: {params.get('n_estimators', '?')} | "
        f"{item_type}: {len(items)}/{params.get('epochs', params.get('max_iter', '?'))}[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    )
    console.print(header)

    # Best epoch/iteration summary
    if epochs:
        # New structure
        best_epoch_idx = data.get('best_epoch', 1) - 1
        if 0 <= best_epoch_idx < len(epochs):
            best_metrics = epochs[best_epoch_idx]['metrics']
            best_panel = Panel(
                f"[bold green]Best Epoch: #{epochs[best_epoch_idx]['epoch']}[/bold green]\n"
                f"Win Rate: [bold]{best_metrics.get('win_rate', 0):.1%}[/bold] | "
                f"W/L: [bold]{best_metrics.get('wl_ratio', 0):.2f}x[/bold] | "
                f"Return: [bold]{best_metrics.get('total_return', 0):.1f}%[/bold] | "
                f"Sharpe: [bold yellow]{best_metrics.get('sharpe_ratio', 0):.2f}[/bold yellow]",
                border_style="green"
            )
            console.print(best_panel)
    else:
        # Old structure
        best = data.get('best_iteration', {})
        if best:
            best_panel = Panel(
                f"[bold green]Best Iteration: #{best.get('iteration', '?')}[/bold green]\n"
                f"Win Rate: [bold]{best.get('win_rate', 0):.1%}[/bold] | "
                f"W/L: [bold]{best.get('wl_ratio', 0):.2f}x[/bold] | "
                f"Return: [bold]{best.get('total_return', 0):.1f}%[/bold] | "
                f"Score: [bold yellow]{best.get('composite_score', 0):.4f}[/bold yellow]",
                border_style="green"
            )
            console.print(best_panel)

    # Metrics progression table
    console.print(f"\n[bold yellow]{item_type} Metrics:[/bold yellow]\n")

    metrics_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    metrics_table.add_column("#", justify="right", style="cyan")
    metrics_table.add_column("Win Rate", justify="right")
    metrics_table.add_column("W/L Ratio", justify="right")
    metrics_table.add_column("Score", justify="right", style="yellow")
    metrics_table.add_column("Return", justify="right")
    metrics_table.add_column("Sharpe", justify="right")
    metrics_table.add_column("Max DD", justify="right")
    metrics_table.add_column("Trades", justify="right")

    for it in items:
        metrics = it['metrics']

        # Color coding based on performance
        wr = metrics['win_rate']
        wr_color = "green" if wr >= 0.75 else "yellow" if wr >= 0.6 else "red"

        # Composite score (may not exist in epoch structure)
        score = metrics.get('composite_score', 0)

        # Use epoch or iteration number
        item_num = str(it.get('epoch', it.get('iteration', '?')))

        metrics_table.add_row(
            item_num,
            f"[{wr_color}]{wr:.1%}[/{wr_color}]",
            f"{metrics['wl_ratio']:.2f}x",
            f"{score:.4f}" if score > 0 else "-",
            f"{metrics['total_return']:.1f}%",
            f"{metrics['sharpe_ratio']:.2f}",
            f"{metrics.get('max_drawdown', 0):.1f}%",
            str(metrics['total_trades'])
        )

    console.print(metrics_table)

    # Sparkline visualizations
    console.print("\n[bold yellow]Progression Charts:[/bold yellow]\n")

    charts_table = Table(show_header=False, box=None, padding=(0, 2))
    charts_table.add_column("Metric", style="bold cyan", width=15)
    charts_table.add_column("Chart", width=40)
    charts_table.add_column("Stats", style="dim")

    # Win Rate sparkline
    wr_spark = create_sparkline([w * 100 for w in win_rates], width=40)
    wr_stats = f"Min: {min(win_rates):.1%} → Max: {max(win_rates):.1%}"
    charts_table.add_row("Win Rate", f"[green]{wr_spark}[/green]", wr_stats)

    # W/L Ratio sparkline
    wl_spark = create_sparkline(wl_ratios, width=40)
    wl_stats = f"Min: {min(wl_ratios):.2f}x → Max: {max(wl_ratios):.2f}x"
    charts_table.add_row("W/L Ratio", f"[blue]{wl_spark}[/blue]", wl_stats)

    # Total Return sparkline (already in percentage form from eval.py)
    ret_spark = create_sparkline(total_returns, width=40)
    ret_stats = f"Min: {min(total_returns):.1f}% → Max: {max(total_returns):.1f}%"
    charts_table.add_row("Total Return", f"[magenta]{ret_spark}[/magenta]", ret_stats)

    # Sharpe Ratio sparkline
    sharpe_spark = create_sparkline(sharpe_ratios, width=40)
    sharpe_stats = f"Min: {min(sharpe_ratios):.2f} → Max: {max(sharpe_ratios):.2f}"
    charts_table.add_row("Sharpe Ratio", f"[yellow]{sharpe_spark}[/yellow]", sharpe_stats)

    # Max Drawdown sparkline (already in percentage form from eval.py)
    dd_spark = create_sparkline([abs(d) for d in max_drawdowns], width=40)
    dd_stats = f"Best: {max(max_drawdowns):.1f}% → Worst: {min(max_drawdowns):.1f}%"
    charts_table.add_row("Max Drawdown", f"[red]{dd_spark}[/red]", dd_stats)

    console.print(charts_table)

    # Show last iteration details if available (only for old structure)
    if iterations and not epochs:
        last_iter = iterations[-1]
        console.print(f"\n[bold yellow]Latest Iteration (#{last_iter['iteration']}):[/bold yellow]\n")

        # Monthly performance
        monthly = last_iter.get('monthly_performance', [])
        if monthly:
            monthly_table = Table(title="Monthly Performance", box=box.SIMPLE)
            monthly_table.add_column("Month", style="cyan")
            monthly_table.add_column("Trades", justify="right")
            monthly_table.add_column("Win Rate", justify="right")
            monthly_table.add_column("Avg PnL", justify="right")

            for month in monthly[-6:]:  # Show last 6 months
                wr = month.get('win_rate', 0)
                wr_color = "green" if wr >= 60 else "yellow" if wr >= 50 else "red"

                monthly_table.add_row(
                    month.get('month', ''),
                    str(month.get('trades', 0)),
                    f"[{wr_color}]{wr:.1f}%[/{wr_color}]",
                    f"{month.get('avg_return', 0):.1f}%"
                )

            console.print(monthly_table)

        # Exit breakdown
        exits = last_iter.get('exit_breakdown', {})
        if exits:
            console.print("\n[bold]Exit Breakdown:[/bold]")
            exit_table = Table(box=box.SIMPLE)
            exit_table.add_column("Exit Type", style="cyan")
            exit_table.add_column("Count", justify="right")

            for exit_type, count in exits.items():
                exit_table.add_row(exit_type, str(count))

            console.print(exit_table)

def list_sessions():
    """List all available training session directories."""
    models_dir = Path('models')

    # Look for new directory structure (training_YYYYMMDD_HHMMSS/)
    session_dirs = sorted([d for d in models_dir.glob('training_*') if d.is_dir()], reverse=True)

    # Fall back to old flat file structure for backward compatibility
    if not session_dirs:
        old_session_files = sorted(models_dir.glob('training_session_*.json'), reverse=True)
        if not old_session_files:
            console.print("[yellow]No training sessions found.[/yellow]")
            return []
        return old_session_files

    # Convert to session.json paths
    session_files = [d / 'session.json' for d in session_dirs if (d / 'session.json').exists()]

    if not session_files:
        console.print("[yellow]No valid training sessions found.[/yellow]")
        return []

    console.print("\n[bold cyan]Available Training Sessions:[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Session ID", style="yellow")
    table.add_column("Started", style="dim")
    table.add_column("Epochs", justify="center")
    table.add_column("Best Win Rate", justify="right")
    table.add_column("Best W/L", justify="right")

    for i, session_file in enumerate(session_files, 1):
        with open(session_file, 'r') as f:
            data = json.load(f)

        params = data.get('parameters', {})

        # Handle both old (iterations) and new (epochs) structure
        epochs = data.get('epochs', [])
        iterations = data.get('iterations', [])
        items = epochs if epochs else iterations
        item_count = len(items)

        # Find best metrics
        if epochs:
            # New structure
            best_epoch_idx = data.get('best_epoch', 1) - 1
            if 0 <= best_epoch_idx < len(epochs):
                best_metrics = epochs[best_epoch_idx]['metrics']
                win_rate = best_metrics.get('win_rate', 0)
                wl_ratio = best_metrics.get('wl_ratio', 0)
            else:
                win_rate = 0
                wl_ratio = 0
        else:
            # Old structure
            best = data.get('best_iteration', {})
            win_rate = best.get('win_rate', 0) if best else 0
            wl_ratio = best.get('wl_ratio', 0) if best else 0

        table.add_row(
            str(i),
            data.get('session_id', 'Unknown'),
            data.get('start_time', 'Unknown')[:19],
            f"{item_count}/{params.get('epochs', params.get('max_iter', '?'))}",
            f"[green]{win_rate:.1%}[/green]" if win_rate > 0 else "-",
            f"{wl_ratio:.2f}x" if wl_ratio > 0 else "-"
        )

    console.print(table)
    return session_files

def main():
    import argparse
    parser = argparse.ArgumentParser(description='View training session in terminal')
    parser.add_argument('session_file', nargs='?', help='Path to training session JSON file')
    parser.add_argument('--list', action='store_true', help='List all available session files')

    args = parser.parse_args()

    if args.list or not args.session_file:
        session_files = list_sessions()

        if not args.session_file and session_files:
            # Interactive selection
            try:
                choice = console.input("\n[bold]Select session number to view (or 0 to exit): [/bold]")
                choice = int(choice)
                if choice > 0 and choice <= len(session_files):
                    args.session_file = str(session_files[choice - 1])
                else:
                    console.print("[dim]Exiting...[/dim]")
                    return
            except (ValueError, KeyboardInterrupt):
                console.print("\n[dim]Exiting...[/dim]")
                return

    if args.session_file:
        console.clear()
        display_session(args.session_file)

if __name__ == "__main__":
    main()
