#!/usr/bin/env python3
"""
Single Stock Analysis Script
Comprehensive analysis of a single ticker with signals, prediction, and recommendation.
"""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

from config import config
from data.storage import DataStorage
from data.fetcher import get_sector_mapping
from ml.model import TradingModel
from ml.features import FeatureEngineer
from signals.combiner import SignalCombiner
from signals.regime_detector import RegimeDetector, MarketRegime
from strategy.position_sizer import PositionSizer

console = Console()


def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize ticker format.
    - Must be 4 letters (e.g., BBCA, TLKM, ASII)
    - Auto-adds .JK suffix for Indonesia Stock Exchange
    """
    # Remove any existing suffix
    base_ticker = ticker.upper().replace(".JK", "").replace("-", "").strip()
    
    # Validate: must be exactly 4 alphabetic characters
    if not re.match(r'^[A-Z]{4}$', base_ticker):
        raise ValueError(f"Invalid ticker '{ticker}'. Must be 4 letters (e.g., BBCA, TLKM, ASII)")
    
    return f"{base_ticker}.JK"


class StockAnalyzer:
    """Comprehensive single stock analyzer."""
    
    def __init__(self):
        self.storage = DataStorage(config.data.db_path)
        self.sector_mapping = get_sector_mapping()
        self.signal_combiner = SignalCombiner(config)
        self.feature_engineer = FeatureEngineer(config.ml)
        self.position_sizer = PositionSizer(config.portfolio)
        self.regime_detector = RegimeDetector()
        
        # Load XGBoost model
        self.model = None
        self.model_trained_date = None
        model_path = os.path.join("models", "global_xgb_champion.pkl")
        if os.path.exists(model_path):
            self.model = TradingModel(config.ml)
            self.model.load(model_path)
            self.model_trained_date = self.model.last_trained
    
    def analyze(self, ticker: str, portfolio_value: Optional[float] = None):
        """Run comprehensive analysis on a single ticker."""
        if portfolio_value is None:
            portfolio_value = config.portfolio.total_value
        
        # Validate and normalize ticker
        try:
            ticker = validate_ticker(ticker)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            return None
        
        # Header
        self._print_header(ticker)
        
        # Fetch data
        price_data = self._fetch_data(ticker)
        if price_data is None:
            return None
        
        # Run all analyses
        basic_info = self._get_basic_info(ticker, price_data)
        technicals = self._get_technicals(price_data)
        ml_prediction = self._get_ml_prediction(price_data)
        signals = self._get_signals(price_data, ml_prediction)  # Pass ML prediction for 60% weight
        regime = self._get_market_regime()
        history = self._get_history(price_data)
        
        # Generate and display comprehensive report
        self._display_report(
            ticker, price_data, basic_info, technicals, 
            ml_prediction, signals, regime, history, portfolio_value
        )
        
        return {
            'ticker': ticker,
            'basic_info': basic_info,
            'technicals': technicals,
            'ml_prediction': ml_prediction,
            'signals': signals,
            'regime': regime
        }
    
    def _print_header(self, ticker: str):
        """Print analysis header."""
        console.print()
        console.print(Panel(
            f"[bold white]{ticker}[/bold white]\n"
            f"[dim]Stock Analysis Report • {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]",
            border_style="blue",
            padding=(0, 2)
        ))
    
    def _fetch_data(self, ticker: str) -> pd.DataFrame:
        """Fetch price data from Yahoo Finance."""
        console.print("\n[dim]Fetching market data...[/dim]", end=" ")
        
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            price_data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if price_data.empty or len(price_data) < 60:
                console.print("[red]✗ Insufficient data[/red]")
                return None
            
            # Standardize columns
            price_data = price_data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            })
            price_data = price_data[['open', 'high', 'low', 'close', 'volume']]
            price_data['ticker'] = ticker
            price_data = price_data.reset_index()
            price_data = price_data.rename(columns={'Date': 'date'})
            price_data['date'] = pd.to_datetime(price_data['date']).dt.tz_localize(None)
            
            console.print(f"[green]✓[/green] {len(price_data)} days loaded")
            return price_data
            
        except Exception as e:
            console.print(f"[red]✗ Failed: {e}[/red]")
            return None
    
    def _get_basic_info(self, ticker: str, df: pd.DataFrame) -> dict:
        """Get basic stock information."""
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
        
        return {
            'sector': self.sector_mapping.get(ticker, "Unknown"),
            'last_price': latest['close'],
            'daily_change': (latest['close'] - prev_close) / prev_close * 100,
            'week_return': (latest['close'] / df.iloc[-5]['close'] - 1) * 100 if len(df) >= 5 else 0,
            'month_return': (latest['close'] / df.iloc[-22]['close'] - 1) * 100 if len(df) >= 22 else 0,
            'ytd_return': (latest['close'] / df.iloc[0]['close'] - 1) * 100,
            'avg_volume': df['volume'].tail(20).mean(),
            'high_52w': df['high'].max(),
            'low_52w': df['low'].min()
        }
    
    def _get_technicals(self, df: pd.DataFrame) -> dict:
        """Calculate technical indicators."""
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = (ema12 - ema26).iloc[-1]
        macd_signal = (ema12 - ema26).ewm(span=9).mean().iloc[-1]
        
        # Moving Averages
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        price = df['close'].iloc[-1]
        
        # Volatility
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        
        return {
            'rsi': rsi,
            'rsi_signal': 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral',
            'macd': macd,
            'macd_signal': 'Bullish' if macd > macd_signal else 'Bearish',
            'price_vs_ma20': (price / sma20 - 1) * 100,
            'price_vs_ma50': (price / sma50 - 1) * 100,
            'ma_signal': 'Above' if price > sma20 and price > sma50 else 'Below' if price < sma20 and price < sma50 else 'Mixed',
            'volatility': volatility,
            'vol_signal': 'High' if volatility > 40 else 'Medium' if volatility > 20 else 'Low'
        }
    
    def _get_ml_prediction(self, df: pd.DataFrame) -> dict:
        """Get ML model prediction."""
        if self.model is None or self.model.model is None:
            return {'probability': None, 'direction': 'N/A', 'confidence': 0, 'error': 'No model trained'}
        
        try:
            # Prepare features
            df_feat = df.copy()
            X = self.feature_engineer.prepare_inference_features(df_feat)
            
            if X.empty:
                return {'probability': None, 'direction': 'N/A', 'confidence': 0, 'error': 'Feature preparation failed'}
            
            # Align features with model's expected features
            if self.model.feature_names:
                # Only keep features that the model was trained on
                available_features = [f for f in self.model.feature_names if f in X.columns]
                missing_features = [f for f in self.model.feature_names if f not in X.columns]
                
                if missing_features:
                    # Add missing features as 0 (safe default)
                    for feat in missing_features:
                        X[feat] = 0
                
                # Reorder columns to match model's expected order
                X = X[self.model.feature_names]
            
            # Get prediction for latest row
            latest_features = X.iloc[-1:].values
            prob = self.model.model.predict_proba(latest_features)[0, 1]
            
            direction = "BULLISH" if prob > 0.55 else "BEARISH" if prob < 0.45 else "NEUTRAL"
            confidence = abs(prob - 0.5) * 200
            
            return {
                'probability': prob,
                'direction': direction,
                'confidence': confidence,
                'error': None
            }
            
        except Exception as e:
            return {'probability': None, 'direction': 'N/A', 'confidence': 0, 'error': str(e)}
    
    def _get_signals(self, df: pd.DataFrame, ml_prediction: dict = None) -> dict:
        """Get trading signals with ML predictions included."""
        try:
            # Create ML predictions Series if we have valid predictions
            ml_predictions = None
            if ml_prediction and ml_prediction.get('probability') is not None:
                # Create a Series with the ML probability for each row
                # The signal combiner expects predictions indexed by DataFrame index
                ml_predictions = pd.Series(
                    [ml_prediction['probability']] * len(df),
                    index=df.index
                )
            
            signals_df = self.signal_combiner.calculate_signals(df, ml_predictions)
            latest = signals_df.iloc[-1]
            
            score = latest.get('composite_score', 0)
            signal = latest.get('signal', 'HOLD')
            
            return {
                'composite_score': score,
                'signal': signal,
                'strength': 'Strong' if abs(score) > 0.5 else 'Moderate' if abs(score) > 0.2 else 'Weak'
            }
        except Exception as e:
            return {'composite_score': 0, 'signal': 'HOLD', 'strength': 'N/A'}
    
    def _get_market_regime(self) -> dict:
        """Get current market regime."""
        try:
            import yfinance as yf
            ihsg = yf.Ticker("^JKSE")
            ihsg_data = ihsg.history(start=datetime.now() - timedelta(days=180), end=datetime.now())
            
            if not ihsg_data.empty:
                prices = ihsg_data['Close']
                regime = self.regime_detector.detect_regime(prices)
                multiplier = self.regime_detector.get_position_multiplier(regime)
                
                return {
                    'regime': regime.value,
                    'multiplier': multiplier
                }
        except Exception:
            pass
        
        return {'regime': 'NORMAL', 'multiplier': 1.0}
    
    def _get_history(self, df: pd.DataFrame) -> dict:
        """Get historical performance stats."""
        returns = df['close'].pct_change().dropna()
        
        # Monthly returns
        df_monthly = df.copy()
        df_monthly['date'] = pd.to_datetime(df_monthly['date'])
        df_monthly = df_monthly.set_index('date')
        monthly_returns = df_monthly['close'].resample('ME').last().pct_change().dropna()
        
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        
        return {
            'daily_win_rate': positive_days / total_days * 100 if total_days > 0 else 0,
            'avg_daily_return': returns.mean() * 100,
            'max_gain': returns.max() * 100,
            'max_loss': returns.min() * 100,
            'monthly_returns': monthly_returns.tail(3).to_dict()
        }
    
    def _display_report(
        self, ticker: str, df: pd.DataFrame, 
        basic: dict, tech: dict, ml: dict, 
        signals: dict, regime: dict, history: dict,
        portfolio_value: float
    ):
        """Display the comprehensive analysis report."""
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION 1: PRICE OVERVIEW
        # ═══════════════════════════════════════════════════════════════════
        console.print("\n[bold blue]━━━ PRICE OVERVIEW ━━━[/bold blue]")
        
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        
        daily_color = "green" if basic['daily_change'] >= 0 else "red"
        week_color = "green" if basic['week_return'] >= 0 else "red"
        month_color = "green" if basic['month_return'] >= 0 else "red"
        ytd_color = "green" if basic['ytd_return'] >= 0 else "red"
        
        table.add_row(
            "Last Price", f"Rp {basic['last_price']:,.0f}",
            "Sector", basic['sector']
        )
        table.add_row(
            "Daily", f"[{daily_color}]{basic['daily_change']:+.2f}%[/{daily_color}]",
            "Week", f"[{week_color}]{basic['week_return']:+.2f}%[/{week_color}]"
        )
        table.add_row(
            "Month", f"[{month_color}]{basic['month_return']:+.2f}%[/{month_color}]",
            "YTD", f"[{ytd_color}]{basic['ytd_return']:+.2f}%[/{ytd_color}]"
        )
        table.add_row(
            "52W High", f"Rp {basic['high_52w']:,.0f}",
            "52W Low", f"Rp {basic['low_52w']:,.0f}"
        )
        console.print(table)
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION 2: TECHNICAL INDICATORS
        # ═══════════════════════════════════════════════════════════════════
        console.print("\n[bold blue]━━━ TECHNICAL INDICATORS ━━━[/bold blue]")
        
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Indicator", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Signal", justify="center")
        
        rsi_color = "green" if tech['rsi_signal'] == 'Oversold' else "red" if tech['rsi_signal'] == 'Overbought' else "yellow"
        macd_color = "green" if tech['macd_signal'] == 'Bullish' else "red"
        ma_color = "green" if tech['ma_signal'] == 'Above' else "red" if tech['ma_signal'] == 'Below' else "yellow"
        vol_color = "red" if tech['vol_signal'] == 'High' else "yellow" if tech['vol_signal'] == 'Medium' else "green"
        
        table.add_row("RSI (14)", f"{tech['rsi']:.1f}", f"[{rsi_color}]{tech['rsi_signal']}[/{rsi_color}]")
        table.add_row("MACD", f"{tech['macd']:.2f}", f"[{macd_color}]{tech['macd_signal']}[/{macd_color}]")
        table.add_row("vs MA20", f"{tech['price_vs_ma20']:+.1f}%", f"[{ma_color}]{tech['ma_signal']}[/{ma_color}]")
        table.add_row("Volatility", f"{tech['volatility']:.1f}%", f"[{vol_color}]{tech['vol_signal']}[/{vol_color}]")
        console.print(table)
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION 3: ML PREDICTION & SIGNALS
        # ═══════════════════════════════════════════════════════════════════
        console.print("\n[bold blue]━━━ ML PREDICTION & SIGNALS ━━━[/bold blue]")
        
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        
        # ML Prediction
        if ml['probability'] is not None:
            ml_color = "green" if ml['direction'] == 'BULLISH' else "red" if ml['direction'] == 'BEARISH' else "yellow"
            table.add_row(
                "ML Direction", f"[{ml_color}]{ml['direction']}[/{ml_color}]",
                "Confidence", f"{ml['confidence']:.0f}%"
            )
            table.add_row(
                "Up Probability", f"{ml['probability']*100:.1f}%",
                "Model Date", self.model_trained_date or "Unknown"
            )
        else:
            table.add_row("ML Prediction", f"[yellow]{ml.get('error', 'Unavailable')}[/yellow]", "", "")
        
        # Signals
        sig_color = "green" if signals['signal'] == 'BUY' else "red" if signals['signal'] == 'SELL' else "yellow"
        table.add_row(
            "Signal", f"[{sig_color}]{signals['signal']}[/{sig_color}]",
            "Strength", signals['strength']
        )
        table.add_row(
            "Composite Score", f"{signals['composite_score']:.2f}",
            "Market Regime", regime['regime']
        )
        console.print(table)
        
        # ═══════════════════════════════════════════════════════════════════
        # SECTION 4: POSITION RECOMMENDATION
        # ═══════════════════════════════════════════════════════════════════
        console.print("\n[bold blue]━━━ RECOMMENDATION ━━━[/bold blue]")
        
        latest = df.iloc[-1]
        entry_price = latest['close']
        
        # Calculate ATR for stops
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        stop_loss = entry_price - (2 * atr)
        take_profit = entry_price + (3 * atr)
        
        # Determine action
        bullish = 0
        bearish = 0
        
        if ml['direction'] == 'BULLISH': bullish += 2
        elif ml['direction'] == 'BEARISH': bearish += 2
        
        if signals['signal'] == 'BUY': bullish += 2
        elif signals['signal'] == 'SELL': bearish += 2
        
        if signals['composite_score'] > 0.2: bullish += 1
        elif signals['composite_score'] < -0.2: bearish += 1
        
        if bullish >= 4:
            action, action_color = "STRONG BUY", "bold green"
        elif bullish >= 2 and bearish < 2:
            action, action_color = "BUY", "green"
        elif bearish >= 4:
            action, action_color = "STRONG SELL", "bold red"
        elif bearish >= 2 and bullish < 2:
            action, action_color = "SELL", "red"
        else:
            action, action_color = "HOLD", "yellow"
        
        # Position sizing
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        position = self.position_sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            stock_volatility=volatility,
            avg_market_volatility=0.20,
            entry_price=entry_price,
            stop_loss=stop_loss,
            confidence=ml['confidence'] / 100 if ml['confidence'] else 0.5,
            market_regime=regime['regime']
        )
        
        console.print(f"\n  [{action_color}]>>> {action} <<<[/{action_color}]")
        
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        
        table.add_row(
            "Entry", f"Rp {entry_price:,.0f}",
            "Shares", f"{position['shares']:,}"
        )
        table.add_row(
            "Stop Loss", f"Rp {stop_loss:,.0f} ({(stop_loss/entry_price-1)*100:+.1f}%)",
            "Position", f"Rp {position['position_value']:,.0f}"
        )
        table.add_row(
            "Take Profit", f"Rp {take_profit:,.0f} ({(take_profit/entry_price-1)*100:+.1f}%)",
            "Portfolio %", f"{position['position_pct']:.1f}%"
        )
        table.add_row(
            "Risk/Reward", f"1 : {abs((take_profit-entry_price)/(entry_price-stop_loss)):.1f}",
            "Risk Amount", f"Rp {position['risk_amount']:,.0f}"
        )
        console.print(table)
        
        # ═══════════════════════════════════════════════════════════════════
        # FOOTER
        # ═══════════════════════════════════════════════════════════════════
        console.print("\n" + "─" * 60)
        console.print("[dim]⚠️  This is not financial advice. Always do your own research.[/dim]")
        console.print()


def main():
    parser = argparse.ArgumentParser(description='Single Stock Analysis')
    parser.add_argument('ticker', type=str, help='Stock ticker (e.g., BBCA or BBCA.JK)')
    
    args = parser.parse_args()
    
    analyzer = StockAnalyzer()
    analyzer.analyze(args.ticker)


if __name__ == "__main__":
    main()
