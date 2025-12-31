#!/usr/bin/env python3
"""
LSTM Hyperparameter Tuning
Grid search for optimal LSTM architecture.
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from sklearn.model_selection import train_test_split
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from ml.features import FeatureEngineer, TBLDataset
from ml.model import LSTMModel
from utils.logger import TimedLogger

console = Console()

def load_data(db_path, limit=None):
    conn = sqlite3.connect(db_path)
    # Load all data (limit parameter kept for future use)
    query = "SELECT date, ticker, open, high, low, close, volume FROM prices ORDER BY date"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df

def train_evaluate_config(X_train, y_train, X_val, y_val, params: Dict, device, progress: Optional[Progress] = None) -> Dict:
    """Train a model with specific params and return best val loss."""
    
    # Create Datasets
    train_dataset = TBLDataset(X_train, y_train)
    val_dataset = TBLDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # Init Model
    model = LSTMModel(
        input_size=5,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        num_classes=3,
        dropout=params['dropout']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    best_val_loss = float('inf')
    early_stop_cnt = 0
    patience = 5
    epochs = 15 # Short epochs for tuning
    
    # Nested Progress Task for Epochs
    epoch_task = None
    if progress:
        epoch_task = progress.add_task(f"[dim]Epochs (H={params['hidden_size']})...[/dim]", total=epochs)
    
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model(X_b)
                loss = criterion(out, y_b)
                val_loss += loss.item()
                _, pred = torch.max(out.data, 1)
                total += y_b.size(0)
                correct += (pred == y_b).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Update Epoch Progress
        if progress and epoch_task:
            progress.update(epoch_task, advance=1, description=f"[dim]Epoch {epoch+1}/{epochs} (L={avg_val_loss:.3f})[/dim]")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= patience:
                break
    
    # Cleanup nested task
    if progress and epoch_task:
        progress.remove_task(epoch_task)
                
    return {
        'params': params,
        'val_loss': best_val_loss,
        'val_acc': val_acc
    }

def main():
    logger = TimedLogger()
    logger.log("Starting LSTM hyperparameter tuning")

    console.print(Panel.fit(
        "[bold cyan]LSTM Hyperparameter Tuning[/bold cyan]\n"
        "[dim]Grid Search for Optimal Architecture[/dim]",
        border_style="cyan"
    ))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    console.print(f"Using device: {device}")
    
    # 1. Load Data (Subset for speed)
    logger.log("Loading data")
    console.print("[yellow]Loading Sample Data...[/yellow]")
    df = load_data(config.data.db_path)
    # Filter 30% of tickers for tuning speed
    tickers = df['ticker'].unique()
    sample_tickers = np.random.choice(tickers, size=int(len(tickers)*0.2), replace=False)
    df = df[df['ticker'].isin(sample_tickers)].copy()
    console.print(f"Using {len(sample_tickers)} tickers for tuning.")
    logger.log(f"Generating sequences for {len(sample_tickers)} tickers")

    # 2. Sequence Gen
    feature_eng = FeatureEngineer(config)
    all_X, all_y = [], []
    
    # Progress Bar for Data Prep
    from rich.progress import track
    for t in track(df['ticker'].unique(), description="Generating Sequences..."):
        tdf = df[df['ticker'] == t].copy()
        if len(tdf) < config.data.window_size + config.ml.tbl_horizon: continue
        X, y = feature_eng.create_sequences(tdf, is_training=True)
        if len(X)>0:
            all_X.append(X)
            all_y.append(y)
    
    if not all_X: return
    
    X_combined = np.concatenate(all_X)
    y_combined = np.concatenate(all_y)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
    logger.log(f"Sequences ready: {len(X_train):,} train, {len(X_val):,} val")

    # 3. Parameter Grid
    grid = [
        {'hidden_size': 8, 'num_layers': 2, 'dropout': 0.0, 'lr': 0.001, 'batch_size': 64},
        {'hidden_size': 16, 'num_layers': 2, 'dropout': 0.0, 'lr': 0.001, 'batch_size': 64},
        {'hidden_size': 32, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001, 'batch_size': 64},
        {'hidden_size': 8, 'num_layers': 4, 'dropout': 0.2, 'lr': 0.001, 'batch_size': 64},
    ]

    logger.log(f"Testing {len(grid)} hyperparameter configurations")
    console.print(f"\n[bold]Testing {len(grid)} configurations...[/bold]")

    results = []
    
    # Progress Bar for Grid Search
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Total Progress", total=len(grid))
        
        for i, params in enumerate(grid):
            # Update desc
            desc = f"Testing [H={params['hidden_size']}, L={params['num_layers']}] ({i+1}/{len(grid)})"
            progress.update(task, description=desc)
            
            # PASS PROGRESS HERE
            res = train_evaluate_config(X_train, y_train, X_val, y_val, params, device, progress=progress)
            results.append(res)
            
            progress.advance(task)
            
    # Summary Table
    results.sort(key=lambda x: x['val_loss'])
    best = results[0]
    logger.log(f"Best config: Hidden={best['params']['hidden_size']}, Loss={best['val_loss']:.4f}")

    table = Table(title="Tuning Results", border_style="blue")
    table.add_column("Hidden", justify="right")
    table.add_column("Layers", justify="right")
    table.add_column("Dropout", justify="right")
    table.add_column("Val Loss", justify="right", style="green")
    table.add_column("Val Acc", justify="right")
    
    for r in results:
        p = r['params']
        is_best = r == best
        style = "bold green" if is_best else None
        
        table.add_row(
            str(p['hidden_size']),
            str(p['num_layers']),
            str(p['dropout']),
            f"{r['val_loss']:.4f}",
            f"{r['val_acc']:.2%}",
            style=style
        )
        
    console.print(table)
    console.print(f"\n[bold green]Best Config Found:[/bold green] {best['params']}")

if __name__ == "__main__":
    main()
