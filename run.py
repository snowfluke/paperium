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
from rich.prompt import Prompt, IntPrompt

console = Console()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    clear_screen()
    console.print(Panel.fit(
        "[bold cyan]Paperium Trading System[/bold cyan]\n"
        "[dim]Dual-Model Intelligence (XGB, GD/SD)[/dim]",
        border_style="cyan"
    ))
    
    console.print("\n[bold yellow]Main Menu:[/bold yellow]")
    console.print("1. [bold green]Morning Ritual[/bold green] (Generate Signals)")
    console.print("2. [bold magenta]Evening Update[/bold magenta] (EOD Retrain & Evaluation)")
    console.print("3. [bold cyan]Model Lab[/bold cyan] (Targeted Model Training)")
    console.print("4. [bold white]Full Strategy Sweep[/bold white] (Global Auto-Train)")
    console.print("0. Exit")
    
    choice = Prompt.ask("\nSelect action", choices=["1", "2", "3", "4", "0"], default="1")
    
    if choice == "1":
        subprocess.run(["uv", "run", "python", "scripts/morning_signals.py"])
    elif choice == "2":
        subprocess.run(["uv", "run", "python", "scripts/eod_retrain.py"])
    elif choice == "3":
        train_menu()
    elif choice == "4":
        subprocess.run(["uv", "run", "python", "scripts/auto_train.py"])
    elif choice == "0":
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)
        
    input("\nPress Enter to return to menu...")
    main_menu()

def train_menu():
    clear_screen()
    console.print(Panel.fit("[bold cyan]Model Training Lab[/bold cyan]", border_style="cyan"))
    
    console.print("\n[bold]Select Model Type:[/bold]")
    console.print("1. XGBoost (Champion)")
    console.print("2. GD/SD (Structural)")
    
    m_choice = Prompt.ask("\nSelect model", choices=["1", "2"], default="1")
    m_map = {"1": "xgboost", "2": "gd_sd"}
    m_type = m_map[m_choice]
    
    target = Prompt.ask("Enter target Win Rate (e.g. 0.85)", default="0.80")
    max_iter = IntPrompt.ask("Enter max optimization iterations", default=3)
    
    cmd = ["uv", "run", "python", "scripts/train_model.py", "--type", m_type, "--target", target, "--max-iter", str(max_iter)]
    
    console.print(f"\n[yellow]Executing: {' '.join(cmd)}[/yellow]\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[dim]Aborted by user.[/dim]")
