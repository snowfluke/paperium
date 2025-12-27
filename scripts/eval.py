#!/usr/bin/env python3
"""
ML-Based Backtest Script
Uses actual XGBoost and Gradient Descent models for prediction

Usage:
    uv run python scripts/ml_backtest.py --start 2024-01-01 --end 2025-09-30
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress

from config import config
from data.storage import DataStorage
from data.fetcher import get_sector_mapping
from ml.model import TradingModel
from signals.screener import Screener

logging.basicConfig(level=logging.WARNING)
console = Console()


class MLBacktest:
    """
    ML-based evaluation using global XGBoost model for multi-day return prediction.
    Target Horizon: 5-Day Forward Return (Day Trading Strategy)
    """
    
    def __init__(self, model_type: str = 'xgboost', retrain: bool = False, custom_model_path: str = None):
        """
        Args:
            model_type: 'xgboost'
            retrain: If True, will train models before backtesting. 
                     If False, will load existing champion models or reject.
            custom_model_path: Optional custom path to model file (for parallel execution)
        """
        self.storage = DataStorage(config.data.db_path)
        self.sector_mapping = get_sector_mapping()
        self.model_type = model_type
        self.retrain = retrain
        self.custom_model_path = custom_model_path
        self.screener = Screener(config)
        
        # Trading parameters
        self.initial_capital = 100_000_000
        self.max_positions = 10
        self.buy_fee = 0.0015
        self.sell_fee = 0.0025
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.08
        self.max_hold_days = 5
        
        # Model (Global XGBoost)
        self.global_xgb = None
    
    def run(self, start_date: str, end_date: str, train_window: int = 252, pre_loaded_data: Optional[pd.DataFrame] = None):
        """
        Run ML backtest with walk-forward optimization.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date  
            train_window: Training window in days
            pre_loaded_data: Optional pre-featured data to speed up iterations
        """
        console.print(Panel.fit(
            f"[bold blue]IHSG ML Evaluation[/bold blue]\n"
            f"[dim]{start_date} to {end_date}[/dim]\n"
            f"[dim]Model: {self.model_type.upper()}[/dim]",
            border_style="blue"
        ))
        
        # Load data
        if pre_loaded_data is not None:
            console.print("\n[yellow]Using pre-loaded featured data...[/yellow]")
            all_data = pre_loaded_data
        else:
            console.print("\n[yellow]Loading data...[/yellow]")
            all_data = self._load_data(start_date, end_date, train_window)
        
        if all_data.empty:
            console.print("[red]No data available[/red]")
            return
        
        try:
            if self.retrain:
                # Pre-train models with strict warning handling
                console.print("[yellow]Training ML models...[/yellow]")
                with warnings.catch_warnings():
                    warnings.simplefilter('error', RuntimeWarning)
                    trained_counts = self._train_models(all_data, start_date, train_window)
                
                if trained_counts['xgb'] == 0:
                    console.print("[red]Critical: XGBoost model was not trained successfuly. Aborting simulation.[/red]")
                    return
            else:
                # Evaluation mode: Load existing models
                console.print("[yellow]Loading existing ML models for evaluation...[/yellow]")
                if not self._load_models():
                    console.print("[bold red]REJECTED: Required models do not exist.[/bold red]")
                    console.print("Please train your models first using 'python scripts/train_model.py'.")
                    return
                
            # Run simulation
            console.print("[yellow]Running simulation...[/yellow]")
            results = self._simulate(all_data, start_date, end_date, is_pre_featured=(pre_loaded_data is not None))
            
            # Display results
            self._display_results(results)
            
        except RuntimeWarning as rw:
            console.print(f"\n[bold red]FATAL: Detected RuntimeWarning during process: {rw}[/bold red]")
            console.print("[red]Aborting simulation to ensure correctness.[/red]")
            return
        except Exception as e:
            console.print(f"\n[bold red]FATAL ERROR: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
            return
        
        return results
    
    def _load_data(self, start_date: str, end_date: str, train_window: int) -> pd.DataFrame:
        """Load price data with buffer for training."""
        # Load ALL history for training
        buffer_start = (
            datetime.strptime(start_date, '%Y-%m-%d') - 
            timedelta(days=1825) # 5 years
        ).strftime('%Y-%m-%d')
        
        data = self.storage.get_prices(start_date=buffer_start, end_date=end_date)
        
        if not data.empty:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values(['ticker', 'date'])
            console.print(f"  ✓ Loaded {len(data)} records for {data['ticker'].nunique()} stocks")
        
        return data
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add internal features and use FeatureEngineer for ML training."""
        # This now just returns the dataframe, as _train_models and _simulate 
        # use FeatureEngineer which calculates everything needed (including target_horizon)
        return df
    
    def _load_models(self) -> bool:
        """Load global models from the models/ directory or custom path."""
        success = True
        
        if self.model_type == 'xgboost':
            # Use custom path if provided, otherwise default
            if self.custom_model_path:
                xgb_path = self.custom_model_path
            else:
                xgb_path = os.path.join("models", "global_xgb_champion.pkl")
            
            if os.path.exists(xgb_path):
                try:
                    self.global_xgb = TradingModel(config.ml)
                    self.global_xgb.load(xgb_path)
                    console.print(f"  ✓ Loaded Global XGBoost model from {xgb_path}")
                except Exception as e:
                    console.print(f"  [red]✗[/red] Failed to load XGBoost: {e}")
                    success = False
            else:
                console.print(f"  [red]✗[/red] XGBoost champion not found at {xgb_path}")
                success = False
        
        return success

    def _train_models(self, all_data: pd.DataFrame, start_date: str, train_window: int):
        """Train global models on pooled historical data."""
        start = pd.to_datetime(start_date)
        
        # Pool training data from all tickers
        pool_X = []
        pool_y = []
        pool_returns = []
        
        console.print(f"  Pooling training data from {all_data['ticker'].nunique()} tickers...")
        
        for ticker in all_data['ticker'].unique():
            ticker_data = all_data[all_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')
            
            # Get data before start_date
            train_data = ticker_data[ticker_data['date'] < start]
            if len(train_data) < 40:
                continue
                
            # Take window
            train_data = train_data.tail(min(train_window, len(train_data)))
            train_data = self._add_features(train_data)
            
            # Use FeatureEngineer to get X, y, returns
            temp_model = TradingModel(config.ml)
            X, y, returns = temp_model.feature_engineer.create_features(
                train_data, 
                target_horizon=temp_model.feature_engineer.target_horizon,
                include_raw_return=True
            )
            pool_X.append(X)
            pool_y.append(y)
            pool_returns.append(returns)
            
        if not pool_X:
            return {'xgb': 0}
            
        X_combined = pd.concat(pool_X)
        y_combined = pd.concat(pool_y)
        ret_combined = pd.concat(pool_returns)
        
        # Clean data for XGBoost (remove any remaining NaN or Inf)
        valid_idx = X_combined.replace([np.inf, -np.inf], np.nan).dropna().index
        X_combined = X_combined.loc[valid_idx]
        y_combined = y_combined.loc[valid_idx]
        ret_combined = ret_combined.loc[valid_idx]
        
        # --- TRAINING REFINEMENT: NOISE DOWNSAMPLING ---
        # Discard 50% of "boring" days (|return| < 0.5%) to reduce noise bias
        noise_mask = (ret_combined.abs() < 0.005)
        keep_mask = ~noise_mask
        
        # Randomly pick 50% of noise to keep
        if noise_mask.any():
            noise_indices = ret_combined[noise_mask].index
            keep_noise_indices = np.random.choice(
                noise_indices, 
                size=len(noise_indices) // 2, 
                replace=False
            )
            keep_mask.loc[keep_noise_indices] = True
            
        X_combined = X_combined[keep_mask]
        y_combined = y_combined[keep_mask]
        ret_combined = ret_combined[keep_mask]
        
        # --- TRAINING REFINEMENT: MAGNITUDE WEIGHTING ---
        # Scale importance by move size: Weight = 1.0 + |return| * 10
        sample_weights = 1.0 + (ret_combined.abs() * 10.0)
        
        console.print(f"  Total training samples after refinement: {len(X_combined)}")
        
        if len(X_combined) < 100:
            console.print("[red]Critical: Not enough training samples pooled.[/red]")
            return {'xgb': 0}

        # Train Global XGBoost
        if self.model_type == 'xgboost':
            try:
                self.global_xgb = TradingModel(config.ml)
                self.global_xgb.feature_names = X_combined.columns.tolist()
                
                # Warm Start: Try to load existing champion
                xgb_champ_path = os.path.join("models", "global_xgb_champion.pkl")
                base_model = None
                if os.path.exists(xgb_champ_path):
                    try:
                        temp_xgb = TradingModel(config.ml)
                        temp_xgb.load(xgb_champ_path)
                        if temp_xgb.model is not None:
                            # Check feature compatibility
                            old_features = set(temp_xgb.feature_names) if temp_xgb.feature_names else set()
                            new_features = set(X_combined.columns.tolist())
                            
                            if old_features == new_features:
                                base_model = temp_xgb.model
                                console.print(f"  → Loaded XGB champion for Warm Start")
                            else:
                                console.print(f"  [yellow]→ Feature mismatch detected (old: {len(old_features)}, new: {len(new_features)}). Training fresh model.[/yellow]")
                    except Exception as load_err:
                        console.print(f"  [yellow]→ Could not load XGB champion for warm start: {load_err}[/yellow]")

                self.global_xgb.model = self.global_xgb._create_model()
                # Remove deprecated param
                if 'use_label_encoder' in self.global_xgb.model.get_params():
                    self.global_xgb.model.set_params(use_label_encoder=False)
                
                if base_model is not None:
                    # Incremental learning with weights
                    self.global_xgb.model.fit(
                        X_combined, y_combined, 
                        sample_weight=sample_weights,
                        xgb_model=base_model.get_booster()
                    )
                else:
                    # Fresh training with weights
                    self.global_xgb.model.fit(
                        X_combined, y_combined, 
                        sample_weight=sample_weights
                    )
                    
                console.print("  ✓ Trained Global XGBoost model")
            except Exception as e:
                console.print(f"  [red]✗[/red] Global XGBoost training failed: {e}")
                self.global_xgb = None

        return {
            'xgb': 1 if self.global_xgb else 0
        }
    
    def _get_prediction(self, ticker: str, df: pd.DataFrame) -> float:
        """Get prediction score using global XGBoost model."""
        if self.global_xgb:
            try:
                prob = self.global_xgb.predict_latest(df)
                return (prob - 0.5) * 2  # Convert to -1 to 1
            except:
                pass
        return 0.0
    
    def _simulate(self, all_data: pd.DataFrame, start_date: str, end_date: str, is_pre_featured: bool = False) -> Dict:
        """Run simulation with pre-calculated features for speed."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if not is_pre_featured:
            # Pre-calculate features for all tickers
            console.print("[yellow]Pre-calculating features for all tickers...[/yellow]")
            ticker_groups = all_data.groupby('ticker')
            processed_data_list = []
            
            with Progress(console=console) as progress:
                task = progress.add_task("[cyan]Processing tickers...", total=all_data['ticker'].nunique())
                for ticker, group in ticker_groups:
                    group = group.sort_values('date')
                    group = self._add_features(group)
                    processed_data_list.append(group)
                    progress.update(task, advance=1)
            
            all_data_feat = pd.concat(processed_data_list).sort_values(['date', 'ticker'])
        else:
            all_data_feat = all_data
        
        # Create a map for quick lookup of a ticker's full processed data
        ticker_data_map = {ticker: group.set_index('date') for ticker, group in all_data_feat.groupby('ticker')}

        # Get all trading dates
        all_dates = sorted(all_data_feat[
            (all_data_feat['date'] >= start) & (all_data_feat['date'] <= end)
        ]['date'].unique())
        
        # Pre-calculate batch scores for ML models (XGBoost and GD part)
        console.print("[yellow]Batch predicting ML scores for all tickers...[/yellow]")
        xgb_scores = {}  # ticker -> Series of scores
        
        for ticker, ticker_df_indexed in ticker_data_map.items():
            ticker_df = ticker_df_indexed.reset_index() # Need original df for feature engineer
            # XGBoost Batch
            if self.global_xgb:
                try:
                    # Use prepare_inference_features for prediction to avoid dropping any rows
                    X = self.global_xgb.feature_engineer.prepare_inference_features(ticker_df)
                    # Align X's index with the original ticker_df's date index for lookup
                    X.index = ticker_df['date']
                    
                    if not X.empty:
                        # Ensure features match model's expected features
                        if self.global_xgb.feature_names:
                            missing = set(self.global_xgb.feature_names) - set(X.columns)
                            for f in missing: X[f] = 0
                            X = X[self.global_xgb.feature_names]
                        
                        probs = self.global_xgb.model.predict_proba(X.values)[:, 1]
                        xgb_scores[ticker] = pd.Series((probs - 0.5) * 2, index=X.index)
                except Exception as e:
                    console.print(f"  [red]✗[/red] XGBoost batch prediction failed for {ticker}: {e}")
        
        # Log prediction stats
        all_xgb = pd.concat(xgb_scores.values()) if xgb_scores else pd.Series()
        
        # Explicitly check for NaNs in scores
        if not all_xgb.empty and all_xgb.isna().any():
            raise ValueError("NaN detected in XGBoost predicted scores.")
            
        if not all_xgb.empty:
            console.print(f"  XGB scores dist: mean={all_xgb.mean():.3f}, max={all_xgb.max():.3f}, min={all_xgb.min():.3f}")
        
        console.print(f"  Simulating {len(all_dates)} trading days...")
        
        # Initialize
        cash = self.initial_capital
        positions = {}
        trades = []
        equity_curve = []
        max_seen_score = -1.0
        
        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Simulating...", total=len(all_dates))
            
            for date in all_dates:
                day_data = all_data_feat[all_data_feat['date'] == date]
                
                # Check exits
                for ticker in list(positions.keys()):
                    ticker_day = day_data[day_data['ticker'] == ticker]
                    if ticker_day.empty:
                        continue
                    
                    row = ticker_day.iloc[0]
                    pos = positions[ticker]
                    
                    # Check stop loss
                    if row['low'] <= pos['stop_loss']:
                        exit_price = pos['stop_loss']
                        cash += pos['shares'] * exit_price * (1 - self.sell_fee)
                        pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                        trades.append({
                            'ticker': ticker, 'entry_date': pos['entry_date'],
                            'exit_date': date, 'pnl_pct': pnl_pct,
                            'exit_reason': 'STOP_LOSS'
                        })
                        del positions[ticker]
                        continue
                    
                    # Check take profit
                    if row['high'] >= pos['take_profit']:
                        exit_price = pos['take_profit']
                        cash += pos['shares'] * exit_price * (1 - self.sell_fee)
                        pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                        trades.append({
                            'ticker': ticker, 'entry_date': pos['entry_date'],
                            'exit_date': date, 'pnl_pct': pnl_pct,
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        del positions[ticker]
                        continue
                    
                    # Check time stop
                    pos['days_held'] = pos.get('days_held', 0) + 1
                    if pos['days_held'] >= self.max_hold_days:
                        exit_price = row['close']
                        cash += pos['shares'] * exit_price * (1 - self.sell_fee)
                        pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                        trades.append({
                            'ticker': ticker, 'entry_date': pos['entry_date'],
                            'exit_date': date, 'pnl_pct': pnl_pct,
                            'exit_reason': 'TIME_STOP'
                        })
                        del positions[ticker]
                
                # Open new positions
                if len(positions) < self.max_positions:
                    candidates = []
                    
                    for ticker in day_data['ticker'].unique():
                        if ticker in positions:
                            continue
                        
                        # Get historical data for prediction (already has features)
                        ticker_hist = all_data_feat[
                            (all_data_feat['ticker'] == ticker) & 
                            (all_data_feat['date'] <= date)
                        ].tail(400) # Increased lookback to satisfy Screener (needs 200)
                        
                        if len(ticker_hist) < 200:
                            continue
                        
                        # Apply Screener logic (Step 2 of workflow)
                        if not self.screener._check_criteria(ticker_hist, ticker):
                            continue
                        
                        # Use batch-predicted XGB score
                        if ticker in xgb_scores and date in xgb_scores[ticker].index:
                            prediction_score = xgb_scores[ticker].loc[date]
                        else:
                            prediction_score = 0.0
                        max_seen_score = max(max_seen_score, prediction_score)
                        
                        # Logic for buy signal (score threshold lowered to 0.1 for debugging)
                        if prediction_score > 0.1:  
                            ticker_day = day_data[day_data['ticker'] == ticker].iloc[0]
                            candidates.append({
                                'ticker': ticker,
                                'score': prediction_score,
                                'close': ticker_day['close'],
                                'atr': ticker_hist['atr'].iloc[-1] if 'atr' in ticker_hist.columns else ticker_day['close'] * 0.02
                            })
                    
                    # Sort by score and take top N
                    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
                    
                    for cand in candidates[:self.max_positions - len(positions)]:
                        # Check sector exposure
                        sector = self.sector_mapping.get(cand['ticker'], 'Unknown')
                        sector_count = sum(1 for p in positions.values() if self.sector_mapping.get(p['ticker'], 'Unknown') == sector)
                        if sector_count >= 3:
                            continue
                            
                        entry_price = cand['close']
                        atr = cand['atr'] if pd.notna(cand['atr']) else entry_price * 0.02
                        
                        position_value = min(cash * 0.12, self.initial_capital * 0.1)
                        shares = int(position_value / entry_price)
                        
                        if shares <= 0:
                            continue
                        
                        cost = shares * entry_price * (1 + self.buy_fee)
                        if cost > cash * 0.95:
                            continue
                        
                        cash -= cost
                        positions[cand['ticker']] = {
                            'ticker': cand['ticker'], # Added for sector check
                            'shares': shares,
                            'entry_price': entry_price,
                            'entry_date': date,
                            'days_held': 0,
                            'stop_loss': entry_price * (1 - max(self.stop_loss_pct, atr / entry_price * 2)),
                            'take_profit': entry_price * (1 + max(self.take_profit_pct, atr / entry_price * 3))
                        }
                
                # Calculate equity
                equity = cash
                for ticker, pos in positions.items():
                    ticker_day = day_data[day_data['ticker'] == ticker]
                    if not ticker_day.empty:
                        equity += pos['shares'] * ticker_day.iloc[0]['close']
                
                equity_curve.append({'date': date, 'equity': equity})
                progress.update(task, advance=1)
        
        console.print(f"  Max combined score seen during simulation: {max_seen_score:.4f}")
        return self._calculate_metrics(trades, equity_curve)
    
    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        if equity_df.empty:
            return {}
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        returns = equity_df['daily_return'].dropna()
        
        rf = 0.05 / 252
        sharpe = np.sqrt(252) * (returns.mean() - rf) / returns.std() if returns.std() > 0 else 0
        
        downside = returns[returns < 0]
        sortino = np.sqrt(252) * (returns.mean() - rf) / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
        
        running_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - running_max) / running_max
        max_dd = drawdown.min()
        
        if not trades_df.empty:
            winners = trades_df[trades_df['pnl_pct'] > 0]
            losers = trades_df[trades_df['pnl_pct'] <= 0]
            win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
            avg_loss = losers['pnl_pct'].mean() if len(losers) > 0 else 0
            
            # Monthly breakdown
            trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
            monthly_groups = trades_df.groupby('exit_month')
            monthly_metrics = []
            
            for month, group in monthly_groups:
                m_winners = group[group['pnl_pct'] > 0]
                m_wr = len(m_winners) / len(group) if len(group) > 0 else 0
                m_ret = group['pnl_pct'].mean() if not group.empty else 0
                monthly_metrics.append({
                    'month': str(month),
                    'win_rate': m_wr * 100,
                    'trades': len(group),
                    'avg_return': m_ret
                })
        else:
            win_rate = avg_win = avg_loss = 0
            monthly_metrics = []
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd * 100,
            'total_trades': len(trades_df),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'monthly_metrics': monthly_metrics,
            'trades': trades_df,
            'equity_curve': equity_df
        }
    
    def _display_results(self, results: Dict):
        """Display backtest results."""
        if not results:
            console.print("[red]No results[/red]")
            return
        
        console.print("\n" + "============================================================\nML EVALUATION RESULTS (XGBOOST)\n============================================================\n")
        
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        ret_style = "green" if results['total_return'] > 0 else "red"
        
        table.add_row("Initial Capital", f"Rp {results['initial_capital']:,.0f}")
        table.add_row("Final Equity", f"Rp {results['final_equity']:,.0f}")
        table.add_row("Total Return", f"[{ret_style}]{results['total_return']:.1f}%[/{ret_style}]")
        table.add_row("", "")
        table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        table.add_row("Sortino Ratio", f"{results['sortino_ratio']:.2f}")
        table.add_row("Max Drawdown", f"[red]{results['max_drawdown']:.1f}%[/red]")
        table.add_row("", "")
        table.add_row("Total Trades", f"{results['total_trades']}")
        table.add_row("Win Rate", f"{results['win_rate']:.1f}%")
        table.add_row("Avg Win", f"[green]+{results['avg_win']:.1f}%[/green]")
        table.add_row("Avg Loss", f"[red]{results['avg_loss']:.1f}%[/red]")
        
        console.print(table)
        
        if results.get('monthly_metrics'):
            console.print("\n[bold]Monthly Performance[/bold]")
            m_table = Table(box=box.MINIMAL, show_header=True)
            m_table.add_column("Month")
            m_table.add_column("Trades", justify="right")
            m_table.add_column("Win Rate", justify="right")
            m_table.add_column("Avg PnL", justify="right")
            
            for m in results['monthly_metrics']:
                wr_style = "green" if m['win_rate'] >= 80 else "yellow" if m['win_rate'] >= 60 else "red"
                m_table.add_row(
                    m['month'],
                    str(m['trades']),
                    f"[{wr_style}]{m['win_rate']:.1f}%[/{wr_style}]",
                    f"{m['avg_return']:+.1f}%"
                )
            console.print(m_table)

        if not results['trades'].empty:
            console.print("\n[bold]Exit Breakdown[/bold]")
            exit_counts = results['trades']['exit_reason'].value_counts()
            for reason, count in exit_counts.items():
                console.print(f"  {reason}: {count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ML-Based Evaluation')
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2025-09-30', help='End date')
    parser.add_argument('--model', choices=['xgboost'], 
                       default='xgboost', help='Model type')
    parser.add_argument('--window', type=int, default=252, help='Training window in trading days')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of models before evaluation')
    parser.add_argument('--model-path', type=str, default=None, help='Custom path to model file (for parallel execution)')
    
    args = parser.parse_args()
    
    bt = MLBacktest(model_type=args.model, retrain=args.retrain, custom_model_path=args.model_path)
    bt.run(args.start, args.end, train_window=args.window)


if __name__ == "__main__":
    main()
