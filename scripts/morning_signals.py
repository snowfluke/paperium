#!/usr/bin/env python3
"""
Morning Signals Script
Run before market open to get today's trading signals

Usage:
    uv run python scripts/morning_signals.py
    
Or if installed:
    uv run morning
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, date
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config import config
from data.fetcher import DataFetcher, get_sector_mapping
from data.storage import DataStorage
from signals.combiner import SignalCombiner
from signals.screener import Screener
from ml.model import EnsembleModel
from strategy.exit_manager import ExitManager
from strategy.position_sizer import PositionSizer
from strategy.position_manager import PositionManager, Position, OrderType, PositionStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()


# Thresholds for order type decision
MARKET_ORDER_THRESHOLD = 0.70  # Lowered slightly
LIMIT_ORDER_THRESHOLD = 0.40   # Lowered slightly
MIN_SIGNAL_SCORE = 0.15        # Minimum score to consider (More inclusive)


class MorningSignals:
    """Generates morning trading signals and manages existing positions."""
    
    def __init__(self):
        self.storage = DataStorage(config.data.db_path)
        self.fetcher = DataFetcher(config.data.stock_universe)
        self.signal_combiner = SignalCombiner(config)
        self.screener = Screener(config)
        
        # Models: Load both if available
        self.models = {}
        from ml.model import TradingModel
        from ml.supply_demand_model import SupplyDemandModel
        
        xgb_path = 'models/global_xgb_champion.pkl'
        if os.path.exists(xgb_path):
            model = TradingModel(config.ml)
            model.load(xgb_path)
            self.models['xgboost'] = model
            logger.info("✓ Loaded XGBoost Champion")
            
        sd_path = 'models/global_sd_champion.pkl'
        if os.path.exists(sd_path):
            model = SupplyDemandModel()
            model.load(sd_path)
            self.models['gd_sd'] = model
            logger.info("✓ Loaded GD/SD Champion")
            
        if not self.models:
            logger.warning("No champion models found! Using indicators only.")

        self.exit_manager = ExitManager(config.exit)
        self.position_sizer = PositionSizer(config.portfolio)
        self.position_manager = PositionManager()
        self.sector_mapping = get_sector_mapping()
    
    def run(self, portfolio_value: float = 100_000_000):
        """
        Run morning signal generation.
        
        Args:
            portfolio_value: Total portfolio value for position sizing
        """
        console.print(Panel.fit(
            f"[bold cyan]IHSG Morning Signals[/bold cyan]\n"
            f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="cyan"
        ))
        
        # Step 0: Interactive Mode Selection
        console.print("\n[bold yellow]Select Mode:[/bold yellow]")
        console.print("1. [bold green]Live Mode[/bold green] (Generate signals & Update positions)")
        console.print("2. [bold magenta]Test Mode[/bold magenta] (View signals only - NO position changes)")
        
        mode_choice = input("\nEnter choice (1 or 2, default=2): ").strip()
        is_live = (mode_choice == '1')
        
        if is_live:
            console.print("[bold red]LOCKED IN LIVE MODE[/bold red]")
        else:
            console.print("[bold blue]RUNNING IN TEST MODE (VIEW ONLY)[/bold blue]")
        
        # Step 1: Update data
        console.print("\n[yellow]Fetching latest market data...[/yellow]")
        self._update_data()
        
        # Step 2: Check existing positions
        console.print("\n[yellow]Checking existing positions...[/yellow]")
        existing_positions = self._review_existing_positions()
        
        # Step 3: Generate new signals
        console.print("\n[yellow]Generating trading signals...[/yellow]")
        new_signals = self._generate_signals(portfolio_value, existing_positions)
        
        # Step 4: Display recommendations
        self._display_recommendations(existing_positions, new_signals)
        
        # Step 5: Save positions ONLY in live mode
        if new_signals and is_live:
            console.print("\n[yellow]Updating position database...[/yellow]")
            self._save_positions(new_signals)
        elif new_signals:
            console.print("\n[blue]Test Mode: Signals displayed but not saved to portfolio.[/blue]")
        
        return new_signals
    
    def _update_data(self):
        """Fetch and update latest price data."""
        # Fetch last 10 days to ensure we have latest data
        data = self.fetcher.fetch_batch(days=10)
        
        if not data.empty:
            count = self.storage.upsert_prices(data)
            console.print(f"  ✓ Updated {count} price records")
        else:
            console.print("  ⚠ No new data available")
    
    def _review_existing_positions(self) -> Dict[str, Dict]:
        """Review and provide recommendations for existing positions."""
        positions = self.position_manager.get_open_positions()
        recommendations = {}
        
        if not positions:
            console.print("  No existing positions")
            return recommendations
        
        # Get latest prices
        all_data = self.storage.get_prices()
        
        for pos in positions:
            ticker = pos.ticker
            ticker_data = all_data[all_data['ticker'] == ticker]
            
            if ticker_data.empty:
                continue
            
            latest = ticker_data.iloc[-1]
            current_price = latest['close']
            high = latest['high']
            low = latest['low']
            
            entry = pos.filled_price or pos.entry_price
            pnl_pct = (current_price - entry) / entry * 100
            
            # Determine recommendation
            action = "HOLD"
            reason = ""
            
            if current_price <= pos.stop_loss:
                action = "SELL (Stop Loss Hit)"
                reason = f"Price {current_price:,.0f} <= SL {pos.stop_loss:,.0f}"
            elif current_price >= pos.take_profit:
                action = "SELL (Take Profit Hit)"
                reason = f"Price {current_price:,.0f} >= TP {pos.take_profit:,.0f}"
            elif pnl_pct < -5:
                action = "REVIEW (Large Loss)"
                reason = f"Floating loss: {pnl_pct:.1f}%"
            elif pnl_pct > 8:
                action = "CONSIDER TRAILING"
                reason = f"Floating profit: {pnl_pct:.1f}%"
            else:
                action = "HOLD"
                reason = f"P&L: {pnl_pct:+.1f}%"
            
            recommendations[ticker] = {
                'position': pos,
                'current_price': current_price,
                'pnl_pct': pnl_pct,
                'action': action,
                'reason': reason
            }
        
        return recommendations
    
    def _generate_signals(
        self, 
        portfolio_value: float,
        existing_positions: Dict[str, Dict]
    ) -> List[Dict]:
        """Generate new trading signals."""
        # Load all data
        all_data = self.storage.get_prices()
        
        if all_data.empty:
            console.print("  [red]No data available. Run data fetch first.[/red]")
            return []
        
        # Calculate signals and ML predictions with progress bar
        data_by_ticker = {}
        ml_predictions = {}
        unique_tickers = all_data['ticker'].unique()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            # Phase 0: Screening
            console.print(f"  [cyan]Running High-Potential Screener...[/cyan]")
            passed_tickers = []
            
            # Group data once for efficiency
            all_grouped = all_data.sort_values('date').groupby('ticker')
            
            screen_task = progress.add_task("Screening candidates...", total=len(unique_tickers))
            
            for ticker in unique_tickers:
                try:
                    df = all_grouped.get_group(ticker)
                    if self.screener._check_criteria(df, ticker):
                        passed_tickers.append(ticker)
                except KeyError:
                    pass
                progress.advance(screen_task)
                
            console.print(f"  [green]✓ {len(passed_tickers)} stocks passed screening out of {len(unique_tickers)}[/green]")
            
            if not passed_tickers:
                return []

            # Phase 1: Technical signals
            sig_task = progress.add_task("Calculating technical signals...", total=len(passed_tickers))
            for ticker in passed_tickers:
                ticker_data = all_grouped.get_group(ticker)
                
                if len(ticker_data) >= config.data.min_data_points:
                    try:
                        signals = self.signal_combiner.calculate_signals(ticker_data)
                        data_by_ticker[ticker] = signals
                    except Exception as e:
                        logger.warning(f"Signal calculation failed for {ticker}: {e}")
                progress.advance(sig_task)
            
            # Phase 2: ML predictions for each model
            all_model_new_signals = []
            
            for model_name, model in self.models.items():
                ml_task = progress.add_task(f"Generating {model_name.upper()} predictions...", total=len(data_by_ticker))
                ml_predictions = {}
                for ticker, ticker_df in data_by_ticker.items():
                    try:
                        if model_name == 'xgboost':
                            prob = model.predict_latest(ticker_df)
                        else: # gd_sd
                            res = model.predict_latest(ticker_df, ticker)
                            prob = (res['combined_score'] / 2) + 0.5
                            
                        ml_predictions[ticker] = pd.Series([prob], index=[ticker_df.index[-1]])
                    except Exception as e:
                        logger.debug(f"{model_name} prediction failed for {ticker}: {e}")
                    progress.advance(ml_task)
                
                # Rank and process for this model
                rankings = self.signal_combiner.rank_stocks(data_by_ticker, ml_predictions)
                if not rankings.empty:
                    # Filter and tag
                    model_buys = rankings[
                        (rankings['signal'] == 'BUY') | (rankings['composite_score'] >= MIN_SIGNAL_SCORE)
                    ].copy()
                    model_buys['model_type'] = model_name
                    all_model_new_signals.append(model_buys)
        
        if not all_model_new_signals:
            return []
            
        # Combine buy signals from all models
        buy_signals = pd.concat(all_model_new_signals).sort_values('composite_score', ascending=False)
        
        # Debug: Print top 5 candidates for confirmation
        if rankings is not None and not rankings.empty:
            console.print("\n[dim]Top 5 Candidates (Debug):[/dim]")
            for _, row in rankings.head(5).iterrows():
                conf_pct = row['composite_score'] * 100
                console.print(f"  [dim]{row['ticker']}: {conf_pct:.1f}%[/dim]")
        
        # Exclude stocks already in portfolio (by SAME model)
        # Allows different models to potentially hold same stock if they both agree
        new_signals = []
        sector_count = {}
        max_per_sector = 3
        max_total_positions = 15 # Increased for multi-model
        max_new_positions = max_total_positions - len(existing_positions)
        
        for _, row in buy_signals.iterrows():
            if len(new_signals) >= max_new_positions:
                break
            
            ticker = row['ticker']
            sector = self.sector_mapping.get(ticker, 'Other')
            
            # Check sector limit
            if sector_count.get(sector, 0) >= max_per_sector:
                continue
            
            # Determine order type based on signal strength
            score = row['composite_score']
            if score >= MARKET_ORDER_THRESHOLD:
                order_type = OrderType.MARKET.value
            else:
                order_type = OrderType.LIMIT.value
            
            # Get entry price and ATR
            entry_price = row['close']
            atr = row.get('atr', entry_price * 0.02)
            
            # For limit orders, set limit price slightly below current
            if order_type == OrderType.LIMIT.value:
                # Limit price at lower band or 1-2% below current
                limit_discount = min(0.02, atr / entry_price)
                limit_price = entry_price * (1 - limit_discount)
            else:
                limit_price = None
            
            # Calculate exit levels
            exit_levels = self.exit_manager.calculate_levels(entry_price, atr)
            
            # Calculate position size
            size_info = self.position_sizer.calculate_position_size(
                portfolio_value=portfolio_value / max_new_positions,
                stock_volatility=row.get('volatility_20', 0.3),
                avg_market_volatility=0.2,
                entry_price=entry_price,
                stop_loss=exit_levels.stop_loss,
                confidence=min(1.0, score + 0.3)
            )
            
            signal = {
                'ticker': ticker,
                'sector': sector,
                'score': score,
                'model_type': row['model_type'],
                'order_type': order_type,
                'entry_price': round(entry_price, 0),
                'limit_price': round(limit_price, 0) if limit_price else None,
                'stop_loss': round(exit_levels.stop_loss, 0),
                'take_profit': round(exit_levels.take_profit, 0),
                'shares': size_info['shares'],
                'position_value': round(size_info['position_value'], 0),
                'risk_reward': exit_levels.reward_ratio
            }
            
            new_signals.append(signal)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        
        return new_signals
    
    def _display_recommendations(
        self, 
        existing_positions: Dict[str, Dict],
        new_signals: List[Dict]
    ):
        """Display recommendations in formatted tables."""
        # Existing Positions Table
        if existing_positions:
            console.print("\n[bold]EXISTING POSITIONS[/bold]")
            
            table = Table(box=box.ROUNDED)
            table.add_column("Ticker", style="cyan")
            table.add_column("Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("P&L %", justify="right")
            table.add_column("Action", style="bold")
            
            for ticker, data in existing_positions.items():
                pos = data['position']
                entry = pos.filled_price or pos.entry_price
                pnl_style = "green" if data['pnl_pct'] > 0 else "red"
                
                table.add_row(
                    ticker.replace('.JK', ''),
                    f"{entry:,.0f}",
                    f"{data['current_price']:,.0f}",
                    f"[{pnl_style}]{data['pnl_pct']:+.1f}%[/{pnl_style}]",
                    data['action']
                )
            
            console.print(table)
        
        # New Signals Table
        console.print("\n[bold]NEW TRADING SIGNALS[/bold]")
        
        if not new_signals:
            console.print("  No new signals meet criteria")
            return
        
        table = Table(box=box.ROUNDED)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Ticker", style="cyan bold")
        table.add_column("Model", justify="center", style="blue")
        table.add_column("Type", justify="center")
        table.add_column("Conf %", justify="right", style="magenta")
        table.add_column("Entry", justify="right")
        table.add_column("Limit", justify="right")
        table.add_column("SL", justify="right", style="red")
        table.add_column("TP", justify="right", style="green")
        table.add_column("Shares", justify="right")
        table.add_column("Value", justify="right")
        
        for i, sig in enumerate(new_signals, 1):
            order_style = "bold green" if sig['order_type'] == 'MARKET' else "yellow"
            # Confidence is score scaled to % (max 100%)
            conf_pct = min(100, max(0, sig['score'] * 100))
            
            table.add_row(
                str(i),
                sig['ticker'].replace('.JK', ''),
                sig['model_type'].upper(),
                f"[{order_style}]{sig['order_type']}[/{order_style}]",
                f"{conf_pct:.1f}%",
                f"{sig['entry_price']:,.0f}",
                f"{sig['limit_price']:,.0f}" if sig['limit_price'] else "-",
                f"{sig['stop_loss']:,.0f}",
                f"{sig['take_profit']:,.0f}",
                f"{sig['shares']:,}",
                f"{sig['position_value']:,.0f}"
            )
        
        console.print(table)
        
        # Summary
        total_value = sum(s['position_value'] for s in new_signals)
        market_orders = sum(1 for s in new_signals if s['order_type'] == 'MARKET')
        limit_orders = len(new_signals) - market_orders
        
        console.print(f"\n[dim]Total: {len(new_signals)} signals | "
                     f"Market: {market_orders} | Limit: {limit_orders} | "
                     f"Value: Rp {total_value:,.0f}[/dim]")
    
    def _save_positions(self, signals: List[Dict]):
        """Save new positions to database."""
        today = date.today().isoformat()
        
        for sig in signals:
            position = Position(
                ticker=sig['ticker'],
                order_type=sig['order_type'],
                model_type=sig['model_type'],
                entry_price=sig['entry_price'],
                limit_price=sig['limit_price'],
                stop_loss=sig['stop_loss'],
                take_profit=sig['take_profit'],
                shares=sig['shares'],
                position_value=sig['position_value'],
                signal_score=sig['score'],
                created_date=today,
                status=PositionStatus.PENDING.value if sig['order_type'] == 'LIMIT' else PositionStatus.OPEN.value,
                filled_date=today if sig['order_type'] == 'MARKET' else None,
                filled_price=sig['entry_price'] if sig['order_type'] == 'MARKET' else None
            )
            
            self.position_manager.create_position(position)
        
        console.print(f"\n[green]✓ Saved {len(signals)} positions[/green]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='IHSG Morning Trading Signals')
    parser.add_argument('--portfolio', type=float, default=100_000_000,
                       help='Total portfolio value (default: 100M IDR)')
    parser.add_argument('--fetch-days', type=int, default=10,
                       help='Days of data to fetch (default: 10)')
    
    args = parser.parse_args()
    
    signals = MorningSignals()
    signals.run(portfolio_value=args.portfolio)


if __name__ == "__main__":
    main()
