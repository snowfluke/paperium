#!/usr/bin/env python3
"""
LSTM Evaluation Script
Evaluates the trained model on test data.
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from sklearn.metrics import classification_report, confusion_matrix
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from ml.model import ModelWrapper
from ml.features import FeatureEngineer, TBLDataset
from utils.logger import TimedLogger

console = Console()

def main():
    logger = TimedLogger()
    parser = argparse.ArgumentParser(description='LSTM Evaluation')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    from rich.progress import track
    
    logger.log("Starting model evaluation")

    console.print(Panel.fit(
        "[bold cyan]Model Evaluation & Backtest[/bold cyan]\n"
        f"[dim]Testing on Data ({args.start} to {args.end or 'Present'})[/dim]",
        border_style="cyan"
    ))
    
    # Load Model
    logger.log("Loading model checkpoint")
    model = ModelWrapper(config)
    epoch, metrics = model.load_checkpoint("models/best_lstm.pt")
    if epoch is None:
         console.print("[yellow]Warning: Could not load model checkpoint. Ensure 'models/best_lstm.pt' exists.[/yellow]")
         return
    console.print(f"[dim]Loaded model from epoch {epoch} (Acc: {metrics.get('val_acc', 0):.1%})[/dim]")
    
    # Load Data
    logger.log(f"Loading test data from {args.start}")
    console.print(f"\n[yellow]Loading Test Data ({args.start}+)...[/yellow]")
    conn = sqlite3.connect(config.data.db_path)
    # Fetch all, we will filter in pandas
    if args.end:
        query = f"SELECT date, ticker, open, high, low, close, volume FROM prices WHERE date >= '{args.start}' AND date <= '{args.end}' ORDER BY date"
    else:
        query = f"SELECT date, ticker, open, high, low, close, volume FROM prices WHERE date >= '{args.start}' ORDER BY date"
    
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    
    console.print(f"Dataset Size: {len(df):,} rows")
    logger.log(f"Data loaded: {len(df):,} rows")

    logger.log("Generating sequences")
    feature_eng = FeatureEngineer(config)
    all_X = []
    all_y = []
    
    tickers = df['ticker'].unique()
    
    # Progress Bar 1: Sequence Generation
    for ticker in track(tickers, description="Generating Sequences..."):
        ticker_df = df[df['ticker'] == ticker].copy()
        if len(ticker_df) < config.data.window_size + config.ml.tbl_horizon:
            continue
            
        X, y = feature_eng.create_sequences(ticker_df, is_training=True)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            
    if not all_X:
        console.print("[red]No valid test sequences found![/red]")
        return
        
    X_test = np.concatenate(all_X)
    y_test = np.concatenate(all_y)
    logger.log(f"Sequences generated: {len(X_test):,} samples")

    logger.log("Running inference on test data")
    dataset = TBLDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Predict
    all_preds = []
    
    model.model.eval()
    with torch.no_grad():
        # Progress Bar 2: Inference
        for X_batch, _ in track(loader, description="Running Inference..."):
            X_batch = X_batch.to(model.device)
            p, _ = model.predict(X_batch)
            all_preds.append(p)
            
    preds = np.concatenate(all_preds)
    logger.log("Inference complete, generating results")

    # ---------------------------------------------------------
    # Visual Reporting
    # ---------------------------------------------------------
    console.print("\n[bold]Evaluation Results[/bold]")
    
    target_names = ['Loss', 'Hold', 'Profit'] # 0, 1, 2
    report_dict = classification_report(y_test, preds, target_names=target_names, output_dict=True)

    # Metris Table
    table = Table(title="Performance Metrics", box=box.SIMPLE)
    table.add_column("Class", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right")
    table.add_column("Count", justify="right")

    for label in target_names:
        metrics = report_dict[label]  # type: ignore

        # Color coding
        prec_col = "green" if metrics['precision'] > 0.5 else "yellow" if metrics['precision'] > 0.3 else "red"  # type: ignore

        table.add_row(
            f"[bold]{label}[/bold]",
            f"[{prec_col}]{metrics['precision']:.1%}[/{prec_col}]",  # type: ignore
            f"{metrics['recall']:.1%}",  # type: ignore
            f"{metrics['f1-score']:.2f}",  # type: ignore
            f"{metrics['support']:,}"  # type: ignore
        )

    # Overall Accuracy
    acc = report_dict['accuracy']  # type: ignore
    table.add_section()
    table.add_row("Total / Avg", "", "", f"[bold]{acc:.1%}[/bold]", f"{len(y_test):,}")
    
    console.print(table)
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, preds)
    
    cm_table = Table(title="Confusion Matrix (Predicted vs Actual)", box=box.ROUNDED)
    cm_table.add_column("Actual \\ Pred", style="dim")
    cm_table.add_column("Loss (Pred)", justify="center")
    cm_table.add_column("Hold (Pred)", justify="center")
    cm_table.add_column("Profit (Pred)", justify="center")
    
    # Row 0: Actual Loss
    cm_table.add_row(
        "[bold red]Loss[/bold red]", 
        f"[green]{cm[0][0]}[/green]", # Correctly predicted loss (Good)
        f"{cm[0][1]}", 
        f"[bold red]{cm[0][2]}[/bold red]" # Dangerous Error: Actual Loss predicted as Profit
    )
    # Row 1: Actual Hold
    cm_table.add_row(
        "Hold", 
        f"{cm[1][0]}", 
        f"[green]{cm[1][1]}[/green]", 
        f"{cm[1][2]}"
    )
    # Row 2: Actual Profit
    cm_table.add_row(
        "[bold green]Profit[/bold green]", 
        f"[red]{cm[2][0]}[/red]", # Missed Opportunity
        f"{cm[2][1]}", 
        f"[bold green]{cm[2][2]}[/bold green]" # Correctly predicted Profit (Great)
    )
    
    console.print(Panel(cm_table, expand=False))

    # Performance Analysis
    # 0=Loss, 1=Hold, 2=Profit
    # Predicted Stats
    pred_profit_signals = int(np.sum(cm[:, 2]))
    
    # Detailed Breakdown
    correct_profit = int(cm[2][2])    # True Positive (Profit -> Profit)
    dangerous_loss = int(cm[0][2])    # Critical Error (Loss -> Profit)
    stagnant_buys = int(cm[1][2])     # Minor Error (Hold -> Profit)
    
    win_rate = correct_profit / pred_profit_signals if pred_profit_signals > 0 else 0
    
    console.print("\n[bold]Model Performance Analysis[/bold]")
    
    summary = Table(show_header=True, box=box.SIMPLE)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="bold")
    summary.add_column("Description", style="dim")
    
    summary.add_row("Total Trades Simulated", f"{pred_profit_signals:,}", "Total 'BUY' signals generated")
    
    wr_color = "green" if win_rate > 0.5 else "yellow"
    summary.add_row("Win Rate", f"[{wr_color}]{win_rate:.1%}[/{wr_color}]", "% of BUY signals that hit target")
    
    summary.add_row("Profit", f"[green]{correct_profit:,}[/green]", "Actual Profit > 3%")
    summary.add_row("Neutral", f"{stagnant_buys:,}", "Price didn't move much (Time Exit)")
    summary.add_row("Loss", f"[red]{dangerous_loss:,}[/red]", "Price hit Stop Loss (-3%)")
    
    console.print(summary)

if __name__ == "__main__":
    main()
