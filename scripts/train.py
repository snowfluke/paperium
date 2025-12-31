#!/usr/bin/env python3
"""
LSTM Training Script
Trains a PyTorch LSTM model using Raw OHLCV + Triple Barrier Labeling.
Features:
- Rich UI with Progress Bars and Metrics
- Session Management (models/training_session_{TIMESTAMP})
- Warm Start / Resume Capability
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import pickle
import argparse
import time
import hashlib
from datetime import datetime
from torch.utils.data import DataLoader
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.layout import Layout
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from ml.model import ModelWrapper
from ml.features import FeatureEngineer, TBLDataset
from utils.logger import TimedLogger

console = Console()

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT date, ticker, open, high, low, close, volume FROM prices ORDER BY date"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_db_version(db_path):
    """Get latest date + row count hash to detect DB changes."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date), COUNT(*) FROM prices")
    max_date, row_count = cursor.fetchone()
    conn.close()
    return f"{max_date}_{row_count}" if max_date else "empty"

def load_or_compute_sequences(df, config, logger, cache_dir=".cache/sequences"):
    """
    Load cached sequences or compute from scratch.
    Cache invalidates when DB version changes.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Create cache key based on DB version + config params
    db_version = get_db_version(config.data.db_path)
    config_hash = hashlib.md5(
        f"{config.data.window_size}_{config.ml.tbl_horizon}_{config.ml.tbl_barrier}".encode()
    ).hexdigest()[:8]
    cache_key = f"sequences_{db_version}_{config_hash}.pkl"
    cache_path = os.path.join(cache_dir, cache_key)

    if os.path.exists(cache_path):
        logger.log(f"[green]✓ Loading cached sequences from {cache_key}[/green]")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Compute sequences from scratch
    logger.log(f"[yellow]Cache miss - computing sequences...[/yellow]")
    feature_eng = FeatureEngineer(config)

    all_X = []
    all_y = []
    tickers = df['ticker'].unique()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"[cyan]Processing {len(tickers)} tickers...",
            total=len(tickers)
        )

        for ticker in tickers:
            ticker_df = df[df['ticker'] == ticker].copy()
            if len(ticker_df) >= config.data.window_size + config.ml.tbl_horizon:
                X, y = feature_eng.create_sequences(ticker_df, is_training=True)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)

            progress.update(task, advance=1)

    logger.log(f"[cyan]Concatenating {len(all_X)} ticker sequences...[/cyan]")
    X_combined = np.concatenate(all_X)
    y_combined = np.concatenate(all_y)
    logger.log(f"[green]✓ Combined {len(X_combined):,} samples[/green]")

    # Save to cache
    logger.log(f"[cyan]Saving to cache...[/cyan]")
    with open(cache_path, 'wb') as f:
        pickle.dump((X_combined, y_combined), f)
    logger.log(f"[green]✓ Cache saved: {cache_key}[/green]")

    return X_combined, y_combined

class TrainingDashboard:
    def __init__(self, epochs, session_id):
        self.epochs = epochs
        self.session_id = session_id

        # Metric history for sparklines/trends (last 10)
        self.history = {
            'train_loss': deque(maxlen=10),
            'val_loss': deque(maxlen=10),
            'val_acc': deque(maxlen=10)
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.start_time = time.time()
        self.current_batch = 0
        self.total_batches = 0

    def get_layout(self, current_epoch, train_loss, val_loss, val_acc, status="Running", batch_info=None):
        if val_loss < self.best_val_loss and val_loss > 0:
            self.best_val_loss = val_loss
            self.best_epoch = current_epoch

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1)
        )

        # Header
        elapsed = time.time() - self.start_time
        header_text = (
            f"[bold cyan]Paperium LSTM Training[/bold cyan] | "
            f"Session: [bold yellow]{self.session_id}[/bold yellow] | "
            f"Time: {int(elapsed//60):02d}:{int(elapsed%60):02d}"
        )
        layout["header"].update(Panel(header_text, style="blue"))

        # Metrics Table
        table = Table(title="Training Metrics", expand=True, border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", justify="right")
        table.add_column("Best", justify="right", style="green")

        table.add_row("Epoch", f"{current_epoch}/{self.epochs}", f"Best: {self.best_epoch}")

        # Show batch progress if training
        if batch_info:
            batch_pct = (batch_info['current'] / batch_info['total'] * 100) if batch_info['total'] > 0 else 0
            batch_str = f"{batch_info['current']}/{batch_info['total']} ({batch_pct:.0f}%)"
            table.add_row("Batch", batch_str, "-")

        table.add_row("Train Loss", f"{train_loss:.4f}", "-")

        best_loss_str = f"{self.best_val_loss:.4f}" if self.best_val_loss != float('inf') else "-"
        color = "green" if val_loss <= self.best_val_loss and val_loss > 0 else "white"
        table.add_row("Val Loss", f"[{color}]{val_loss:.4f}[/{color}]", best_loss_str)

        table.add_row("Val Accuracy", f"{val_acc:.2%}", "-")
        table.add_row("Status", status, "")

        layout["body"].update(Panel(table, title="Active Monitoring", border_style="blue"))
        return layout

def main():
    parser = argparse.ArgumentParser(description='Paperium LSTM Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from (e.g. models/session_X/last.pt)')
    parser.add_argument('--retrain', action='store_true', help='Continue training from best_lstm.pt if it exists')
    args = parser.parse_args()

    # Initialize timed logger
    logger = TimedLogger()

    # Determine checkpoint path
    checkpoint_path = None
    if args.retrain:
        if os.path.exists("models/best_lstm.pt"):
            checkpoint_path = "models/best_lstm.pt"
            logger.log(f"[yellow]Retrain mode: Using existing model at {checkpoint_path}[/yellow]")
        else:
            logger.log(f"[yellow]Retrain mode: No existing model found, starting fresh[/yellow]")
    elif args.resume:
        checkpoint_path = args.resume

    # 0. Setup Session
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("models", f"training_session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)

    logger.log(f"[bold cyan]Initializing Session: {session_id}[/bold cyan]")

    # 1. Load Data (with sequence caching)
    with console.status("[yellow]Loading market data from SQLite...[/yellow]"):
        df = load_data(config.data.db_path)

    logger.log(f"[green]✓ Loaded {len(df):,} price records[/green]")

    # Load or compute sequences (uses cache if DB hasn't changed)
    X_combined, y_combined = load_or_compute_sequences(df, config, logger)

    logger.log(f"[green]✓ Sequences ready:[/green] {len(X_combined):,} total samples")

    with console.status("[yellow]Splitting train/validation sets...[/yellow]"):
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, shuffle=True)

    logger.log(f"[green]✓ Train/val split complete[/green]")

    with console.status("[yellow]Creating data loaders...[/yellow]"):
        train_dataset = TBLDataset(X_train, y_train)
        val_dataset = TBLDataset(X_val, y_val)

        # Optimize for GPU: parallel loading + pinned memory for faster transfer
        num_workers = 2 if torch.cuda.is_available() else 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.ml.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.ml.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )

    logger.log(f"[green]✓ Data Loaded:[/green] {len(X_train):,} train samples, {len(X_val):,} val samples")

    # 2. Init Model
    logger.log(f"[cyan]Initializing LSTM model...[/cyan]")
    model_wrapper = ModelWrapper(config)
    start_epoch = 0

    # Resume logic (either from --retrain or --resume flag)
    if checkpoint_path:
        logger.log(f"[yellow]Loading checkpoint from {checkpoint_path}...[/yellow]")
        start_epoch, prev_metrics = model_wrapper.load_checkpoint(checkpoint_path)
        if start_epoch is not None:
            logger.log(f"[bold yellow]✓ Resuming from epoch {start_epoch}[/bold yellow]")
            start_epoch += 1 # Continue from next epoch
        else:
            logger.log(f"[bold red]Failed to load checkpoint, starting fresh[/bold red]")
            start_epoch = 0
    else:
        logger.log(f"[green]Starting fresh training (no checkpoint loaded)[/green]")

    logger.log(f"[green]✓ Model ready on device: {model_wrapper.device}[/green]")

    # 3. Training Loop
    logger.log(f"[bold cyan]Starting training for {args.epochs} epochs...[/bold cyan]")
    dashboard = TrainingDashboard(args.epochs, session_id)
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    )
    
    # Live Display
    with Live(dashboard.get_layout(start_epoch, 0, 0, 0), refresh_per_second=4, console=console) as live:

        patience = 10
        patience_counter = 0
        best_val_loss = float('inf')

        for epoch in range(start_epoch, args.epochs):
            # Progress callback for batch-level updates
            current_train_loss = [0.0]  # Mutable container for closure
            current_train_acc = [0.0]

            def batch_progress(batch_num, total_batches, loss, acc):
                current_train_loss[0] = loss
                current_train_acc[0] = acc
                batch_info = {'current': batch_num, 'total': total_batches}
                live.update(dashboard.get_layout(
                    epoch+1, loss, 0, 0,
                    f"Training... (Batch {batch_num}/{total_batches})",
                    batch_info=batch_info
                ))

            # Train Epoch with live batch progress
            train_loss, train_acc = model_wrapper.train_one_epoch(train_loader, progress_callback=batch_progress)

            # Validation progress callback
            def val_progress(batch_num, total_batches, loss, acc):
                batch_info = {'current': batch_num, 'total': total_batches}
                live.update(dashboard.get_layout(
                    epoch+1, train_loss, loss, acc,
                    f"Validating... (Batch {batch_num}/{total_batches})",
                    batch_info=batch_info
                ))

            # Evaluate with live batch progress
            val_loss, val_acc = model_wrapper.evaluate(val_loader, progress_callback=val_progress)

            # Update Dashboard with final epoch metrics
            live.update(dashboard.get_layout(epoch+1, train_loss, val_loss, val_acc, "Saving checkpoint..."))

            # Checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                model_wrapper.save_checkpoint(f"{session_dir}/best.pt", epoch, {"val_loss": val_loss, "val_acc": val_acc})
                # Also verify/copy to global best for inference
                model_wrapper.save_checkpoint("models/best_lstm.pt", epoch, {"val_loss": val_loss, "val_acc": val_acc})
            else:
                patience_counter += 1

            if patience_counter >= patience:
                live.update(dashboard.get_layout(epoch+1, train_loss, val_loss, val_acc, "Early Stopping"))
                break

            model_wrapper.save_checkpoint(f"{session_dir}/last.pt", epoch, {"val_loss": val_loss, "val_acc": val_acc})

    console.print("")  # Blank line for spacing
    logger.log(f"[bold green]✓ Training Complete![/bold green]")
    logger.log(f"[green]Best Val Loss:[/green] {best_val_loss:.4f}")
    logger.log(f"[green]Session saved to:[/green] {session_dir}")

if __name__ == "__main__":
    main()
