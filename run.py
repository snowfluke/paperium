#!/usr/bin/env python3
"""
Paperium Unified Runner
Provides an easy-to-use CLI to run all project workflows.
"""
import os
import sys
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm

console = Console()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    clear_screen()
    console.print(Panel.fit(
        "[bold cyan]Paperium Trading System[/bold cyan]\n"
        "[dim]XGBoost Intelligence[/dim]",
        border_style="cyan"
    ))
    
    console.print("\n[bold yellow]Main Menu:[/bold yellow]")
    console.print("1. [bold green]Morning Ritual[/bold green] (Generate Signals)")
    console.print("2. [bold magenta]Evening Update[/bold magenta] (EOD Retrain & Evaluation)")
    console.print("3. [bold cyan]Model Training[/bold cyan] (Customizable)")
    console.print("4. [bold blue]Evaluation[/bold blue] (Backtest)")
    console.print("5. [bold white]Stock Analysis[/bold white] (Single Ticker Deep Dive)")
    console.print("0. Exit")
    
    choice = Prompt.ask("\nSelect action", choices=["1", "2", "3", "4", "5", "0"], default="1")
    
    if choice == "1":
        custom_capital = IntPrompt.ask("Extra Money / Free Capital to allocate (IDR)", default=0)
        cmd = ["uv", "run", "python", "scripts/morning_signals.py"]
        if custom_capital > 0:
            cmd.extend(["--custom-capital", str(custom_capital)])
        subprocess.run(cmd)
    elif choice == "2":
        subprocess.run(["uv", "run", "python", "scripts/eod_retrain.py"])
    elif choice == "3":
        train_menu()
    elif choice == "4":
        eval_menu()
    elif choice == "5":
        analyze_menu()
    elif choice == "0":
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)
        
    input("\nPress Enter to return to menu...")
    main_menu()

def analyze_menu():
    clear_screen()
    console.print(Panel.fit("[bold cyan]Single Stock Analysis[/bold cyan]", border_style="cyan"))
    
    console.print("\n[dim]Comprehensive analysis of a single stock[/dim]\n")
    
    ticker = Prompt.ask("Enter ticker (e.g., BBCA)")
    portfolio = IntPrompt.ask("Portfolio value (IDR)", default=100_000_000)
    
    cmd = ["uv", "run", "python", "scripts/analyze.py", ticker, "--portfolio", str(portfolio)]
    
    console.print(f"\n[yellow]Executing: {' '.join(cmd)}[/yellow]\n")
    subprocess.run(cmd)

def train_menu():
    clear_screen()
    console.print(Panel.fit("[bold cyan]Model Training Lab[/bold cyan]", border_style="cyan"))
    
    # Customizable parameters
    console.print("\n[dim]Configure training parameters:[/dim]\n")
    
    # Show current champion info
    try:
        import json
        with open('models/champion_metadata.json', 'r') as f:
            metadata = json.load(f)
        current_wr = metadata.get('xgboost', {}).get('win_rate', 0)
        current_wl = metadata.get('xgboost', {}).get('wl_ratio', 0)
        current_score = metadata.get('xgboost', {}).get('combined_score', current_wr)
        console.print(f"[cyan]Current Champion:[/cyan] WR={current_wr:.1%}, W/L={current_wl:.2f}x, Score={current_score:.1%}\n")
    except:
        pass
    
    target = Prompt.ask("Target Combined Score (Win Rate + W/L Ratio, 0.0-1.0)", default="0.85")
    days = Prompt.ask("Evaluation days (number or 'max')", default="365")
    train_window = Prompt.ask("Training window (number or 'max')", default="max")
    max_iter = IntPrompt.ask("Max optimization iterations", default=10)
    
    # Gen-5 hyperparameters
    console.print("\n[yellow]Gen-5 Hyperparameters (leave blank for defaults):[/yellow]")
    max_depth = Prompt.ask("  Max tree depth", default="6")
    n_estimators = Prompt.ask("  Number of estimators", default="150")
    learning_rate = Prompt.ask("  Learning rate", default="0.1")
    
    force = Confirm.ask("\nReplace champion if better?", default=True)
    
    cmd = ["uv", "run", "python", "scripts/train.py",
           "--target", target,
           "--days", days,
           "--train-window", train_window,
           "--max-iter", str(max_iter),
           "--max-depth", max_depth,
           "--n-estimators", n_estimators,
           "--learning-rate", learning_rate]
    
    if force:
        cmd.append("--force")
    
    console.print(f"\n[yellow]Executing: {' '.join(cmd)}[/yellow]\n")
    subprocess.run(cmd)

def eval_menu():
    clear_screen()
    console.print(Panel.fit("[bold cyan]Evaluation Lab[/bold cyan]", border_style="cyan"))
    
    console.print("\n[dim]Configure evaluation parameters:[/dim]\n")
    
    start_date = Prompt.ask("Start date (YYYY-MM-DD)", default="2024-01-01")
    end_date = Prompt.ask("End date (YYYY-MM-DD)", default="2025-09-30")
    window = IntPrompt.ask("Training window (trading days)", default=252)
    retrain = Confirm.ask("Retrain model before evaluation?", default=False)
    
    cmd = ["uv", "run", "python", "scripts/eval.py", 
           "--start", start_date, 
           "--end", end_date,
           "--window", str(window)]
    
    if retrain:
        cmd.append("--retrain")
    
    console.print(f"\n[yellow]Executing: {' '.join(cmd)}[/yellow]\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[dim]Aborted by user.[/dim]")
