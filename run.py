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

def run_script(script_path, *args):
    """Helper to run scripts with uv."""
    cmd = ["uv", "run", "python", script_path]
    cmd.extend(args)
    console.print(f"\n[yellow]Executing: {' '.join(cmd)}[/yellow]\n")
    subprocess.run(cmd)

def main_menu():
    clear_screen()
    console.print(Panel.fit(
        "[bold cyan]Paperium Trading System[/bold cyan]\n"
        "[dim]LSTM + Triple Barrier Labeling Algorithm[/dim]",
        border_style="cyan"
    ))
    
    console.print("\n[bold]Main Menu:[/bold]")
    console.print("0. Initial Setup & Data Prep (Mandatory)")
    console.print("1. IDX Stock Prediction (Generate Signals)")
    console.print("2. Model Training (LSTM)")
    console.print("3. Evaluation (Backtest)")
    console.print("4. Hyperparameter Tuning (Optimize LSTM)")
    console.print("X. Exit")

    choice = Prompt.ask("\nSelect action", choices=["0", "1", "2", "3", "4", "5", "X", "x"], default="1")
    
    if choice == "0":
        setup_menu()
    elif choice == "1":
        fetch_latest = Confirm.ask("Fetch latest data from Yahoo Finance?", default=False)
        custom_capital = IntPrompt.ask("Capital to allocate (IDR, 0=show all signals)", default=0)
        num_stock = IntPrompt.ask("Number of stocks to trade (0=show all)", default=0)
        cmd = ["scripts/signals.py"]
        if fetch_latest:
            cmd.append("--fetch-latest")
        if custom_capital > 0:
            cmd.extend(["--capital", str(custom_capital)])
        if num_stock > 0:
            cmd.extend(["--num-stock", str(num_stock)])
        run_script(*cmd)
    elif choice == "2":
        train_menu() # Now interactive/rich
    elif choice == "3":
        eval_menu()
    elif choice == "4":
        tune_menu()
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
    console.print("2. [bold magenta]Sync Stock Data[/bold magenta] (Fetch historical data)")
    console.print("3. [bold green]Download IHSG Index[/bold green] (For market context)")
    console.print("4. [bold red]Clear Cache[/bold red] (Delete .cache folder)")
    console.print("B. Back to Main Menu")

    choice = Prompt.ask("\nSelect setup action", choices=["1", "2", "3", "4", "B", "b"], default="1")

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

def tune_menu():
    clear_screen()
    console.print(Panel.fit("[bold yellow]Hyperparameter Tuning[/bold yellow]", border_style="yellow"))
    console.print("\n[dim]Optimize LSTM architecture (Hidden Size, Layers, Dropout)[/dim]\n")
    
    cmd = ["uv", "run", "python", "scripts/tune_lstm.py"]
    console.print(f"[yellow]Launching tuner...[/yellow]\n")
    subprocess.run(cmd)

def train_menu():
    clear_screen()
    console.print(Panel.fit("[bold cyan]LSTM Model Training[/bold cyan]", border_style="cyan"))

    # Customizable parameters
    console.print("\n[dim]Configure training parameters:[/dim]\n")

    # Check if existing model exists
    model_exists = os.path.exists("models/best_lstm.pt")

    if model_exists:
        console.print("[yellow]⚠ Existing model detected:[/yellow] models/best_lstm.pt\n")
        training_mode = Prompt.ask(
            "Training mode",
            choices=["fresh", "retrain"],
            default="retrain"
        )
        console.print()

        if training_mode == "fresh":
            console.print("[bold yellow]Fresh Training:[/bold yellow] Starting new model from scratch")
        else:
            console.print("[bold cyan]Retrain Mode:[/bold cyan] Continuing from existing model")
    else:
        console.print("[dim]No existing model found - will start fresh training[/dim]\n")
        training_mode = "fresh"

    epochs = IntPrompt.ask("Training epochs", default=50)

    cmd = ["uv", "run", "python", "scripts/train.py", "--epochs", str(epochs)]

    if training_mode == "retrain":
        cmd.append("--retrain")

    console.print(f"\n[yellow]Executing: {' '.join(cmd)}[/yellow]\n")
    subprocess.run(cmd)


def eval_menu():
    clear_screen()
    console.print(Panel.fit("[bold cyan]Evaluation Lab[/bold cyan]", border_style="cyan"))
    
    console.print("\n[dim]Configure evaluation parameters:[/dim]\n")
    
    start_date = Prompt.ask("Start date (YYYY-MM-DD)", default="2024-01-01")
    end_date = Prompt.ask("End date (YYYY-MM-DD)", default="2025-09-30")
    
    cmd = ["uv", "run", "python", "scripts/eval.py", 
           "--start", start_date, 
           "--end", end_date]
    
    console.print(f"\n[yellow]Executing: {' '.join(cmd)}[/yellow]\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[dim]Aborted by user.[/dim]")
