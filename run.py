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
    console.print("0. [bold yellow]Initial Setup & Data Prep[/bold yellow] (Mandatory)")
    console.print("1. [bold green]Morning Ritual[/bold green] (Generate Signals)")
    console.print("2. [bold magenta]Evening Update[/bold magenta] (EOD Retrain & Evaluation)")
    console.print("3. [bold cyan]Model Training[/bold cyan] (Customizable)")
    console.print("4. [bold blue]Evaluation[/bold blue] (Backtest)")
    console.print("5. [bold white]Stock Analysis[/bold white] (Single Ticker Deep Dive)")
    console.print("6. [bold yellow]Training Sessions[/bold yellow] (View Progress)")
    console.print("X. Exit")

    choice = Prompt.ask("\nSelect action", choices=["1", "2", "3", "4", "5", "6", "0", "X", "x"], default="1")
    
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
    elif choice == "6":
        view_training_menu()
    elif choice == "0":
        setup_menu()
    elif choice.upper() == "X":
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)
        
    input("\nPress Enter to return to menu...")
    main_menu()

def setup_menu():
    clear_screen()
    console.print(Panel.fit("[bold yellow]Initial Setup & Data Prep[/bold yellow]", border_style="yellow"))
    
    console.print("\n[dim]Prepare your environment and sync market data[/dim]\n")
    console.print("1. [bold cyan]Clean Universe[/bold cyan] (Filter illiquid/suspended stocks)")
    console.print("2. [bold magenta]Sync Stock Data[/bold magenta] (Fetch 5 years of daily history)")
    console.print("3. [bold green]Download IHSG Index[/bold green] (For crash detection)")
    console.print("4. [bold yellow]Analyze Hour-0 Patterns[/bold yellow] (Fetch hourly & calculate metrics)")
    console.print("5. [bold red]Clear Cache[/bold red] (Delete .cache folder)")
    console.print("B. Back to Main Menu")


    choice = Prompt.ask("\nSelect setup action", choices=["1", "2", "3", "4", "5", "B", "b"], default="1")

    if choice == "1":
        console.print("\n[yellow]Running: uv run python scripts/clean_universe.py[/yellow]\n")
        subprocess.run(["uv", "run", "python", "scripts/clean_universe.py"])
    elif choice == "2":
        console.print("\n[yellow]Running: uv run python scripts/sync_data.py[/yellow]\n")
        subprocess.run(["uv", "run", "python", "scripts/sync_data.py"])
    elif choice == "3":
        days = IntPrompt.ask("Days of IHSG history to download", default=1825)
        console.print(f"\n[yellow]Running: uv run python scripts/download_ihsg.py --days {days}[/yellow]\n")
        subprocess.run(["uv", "run", "python", "scripts/download_ihsg.py", "--days", str(days)])
    elif choice == "4":
        from rich.prompt import Confirm
        use_all = Confirm.ask("Analyze all tickers from universe?", default=True)

        if use_all:
            days = IntPrompt.ask("Days of hourly data to fetch (max 60)", default=60)
            console.print(f"\n[yellow]Running: uv run python scripts/analyze_hour0.py --days {days}[/yellow]\n")
            console.print("[dim]⏱ This will fetch hourly data with rate limiting (all stocks = ~5 minutes)[/dim]\n")
            subprocess.run(["uv", "run", "python", "scripts/analyze_hour0.py", "--days", str(days)])
        else:
            stocks = IntPrompt.ask("Number of stocks to analyze", default=200)
            days = IntPrompt.ask("Days of hourly data to fetch (max 60)", default=60)
            console.print(f"\n[yellow]Running: uv run python scripts/analyze_hour0.py --stocks {stocks} --days {days}[/yellow]\n")
            console.print(f"[dim]⏱ This will fetch hourly data with rate limiting ({stocks} stocks = ~{stocks // 10 * 3 // 60} minutes)[/dim]\n")
            subprocess.run(["uv", "run", "python", "scripts/analyze_hour0.py", "--stocks", str(stocks), "--days", str(days)])
    elif choice == "5":
        import shutil
        if os.path.exists(".cache"):
            from rich.prompt import Confirm
            confirm = Confirm.ask("\n[red]⚠  Delete .cache folder? This cannot be undone.[/red]")
            if confirm:
                shutil.rmtree(".cache")
                console.print("[green]✓ Cache cleared[/green]")
            else:
                console.print("[dim]Cancelled[/dim]")
        else:
            console.print("[dim].cache folder does not exist[/dim]")
    elif choice.upper() == "B":
        return

def analyze_menu():
    clear_screen()
    console.print(Panel.fit("[bold cyan]Single Stock Analysis[/bold cyan]", border_style="cyan"))
    
    console.print("\n[dim]Comprehensive analysis of a single stock[/dim]\n")
    
    ticker = Prompt.ask("Enter ticker (e.g., BBCA)")
    portfolio = IntPrompt.ask("Portfolio value (IDR)", default=100_000_000)
    
    cmd = ["uv", "run", "python", "scripts/analyze.py", ticker, "--portfolio", str(portfolio)]
    
    console.print(f"\n[yellow]Executing: {' '.join(cmd)}[/yellow]\n")
    subprocess.run(cmd)

def view_training_menu():
    clear_screen()
    console.print(Panel.fit("[bold yellow]Training Session Viewer[/bold yellow]", border_style="yellow"))

    console.print("\n[dim]View training session progression in terminal[/dim]\n")

    cmd = ["uv", "run", "python", "scripts/view_training.py"]

    console.print(f"[yellow]Launching session viewer...[/yellow]\n")
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
            try:
                metadata = json.load(f)
            except (json.JSONDecodeError, ValueError):
                metadata = {}
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
    
    use_gpu = Confirm.ask("\nUse GPU acceleration?", default=True)
    force = Confirm.ask("Replace champion if better?", default=True)
    
    cmd = ["uv", "run", "python", "scripts/train.py",
           "--target", target,
           "--days", days,
           "--train-window", train_window,
           "--max-iter", str(max_iter)]
    
    if force:
        cmd.append("--force")
    
    if use_gpu:
        cmd.append("--gpu")
    
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
