#!/usr/bin/env python3
"""
Single Stock Analysis Script
Comprehensive analysis of a single ticker with signals, prediction, and recommendation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.markdown import Markdown

from config import config
from data.storage import DataStorage
from data.fetcher import DataFetcher, get_sector_mapping
from ml.model import TradingModel
from ml.features import FeatureEngineer
from signals.combiner import SignalCombiner
from signals.regime_detector import RegimeDetector, MarketRegime
from signals.screener import Screener
from strategy.position_sizer import PositionSizer

console = Console()


class StockAnalyzer:
    """Comprehensive single stock analyzer."""
    
    def __init__(self):
        self.storage = DataStorage(config.data.db_path)
        self.fetcher = DataFetcher(config.data.stock_universe)
        self.sector_mapping = get_sector_mapping()
        self.signal_combiner = SignalCombiner(config)
        self.feature_engineer = FeatureEngineer(config.ml)
        self.screener = Screener(config)
        self.position_sizer = PositionSizer(config.portfolio)
        self.regime_detector = RegimeDetector()
        
        # Load XGBoost model
        self.model = None
        model_path = os.path.join("models", "global_xgb_champion.pkl")
        if os.path.exists(model_path):
            self.model = TradingModel(config.ml)
            self.model.load(model_path)
    
    def analyze(self, ticker: str, portfolio_value: float = 100_000_000):
        """Run comprehensive analysis on a single ticker."""
        console.print(Panel.fit(
            f"[bold cyan]Stock Analysis: {ticker}[/bold cyan]\n"
            f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="cyan"
        ))
        
        # 1. Fetch latest data using yfinance directly
        console.print("\n[yellow]📊 Fetching Data...[/yellow]")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            price_data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if price_data.empty:
                console.print(f"[red]No data found for {ticker}[/red]")
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
            
        except Exception as e:
            console.print(f"[red]Failed to fetch data: {e}[/red]")
            return None
        
        # 2. Basic Info
        self._display_basic_info(ticker, price_data)
        
        # 3. Technical Analysis
        self._display_technical_analysis(price_data)
        
        # 4. ML Prediction
        prediction = self._get_ml_prediction(price_data)
        
        # 5. Signal Analysis
        signals = self._analyze_signals(price_data)
        
        # 6. Market Regime
        regime = self._get_market_regime()
        
        # 7. Historical Performance
        self._display_historical_performance(price_data)
        
        # 8. Position Recommendation
        self._generate_recommendation(
            ticker, price_data, prediction, signals, regime, portfolio_value
        )
        
        return {
            'ticker': ticker,
            'prediction': prediction,
            'signals': signals,
            'regime': regime
        }
    
    def _display_basic_info(self, ticker: str, df: pd.DataFrame):
        """Display basic stock information."""
        console.print("\n[bold]Basic Information[/bold]")
        
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
        daily_change = (latest['close'] - prev_close) / prev_close * 100
        
        # Calculate stats
        week_return = (latest['close'] / df.iloc[-5]['close'] - 1) * 100 if len(df) >= 5 else 0
        month_return = (latest['close'] / df.iloc[-22]['close'] - 1) * 100 if len(df) >= 22 else 0
        ytd_return = (latest['close'] / df.iloc[0]['close'] - 1) * 100
        
        avg_volume = df['volume'].tail(20).mean()
        
        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Sector", self.sector_mapping.get(ticker, "Unknown"))
        table.add_row("Last Price", f"Rp {latest['close']:,.0f}")
        table.add_row("Daily Change", f"[{'green' if daily_change >= 0 else 'red'}]{daily_change:+.2f}%[/]")
        table.add_row("Week Return", f"[{'green' if week_return >= 0 else 'red'}]{week_return:+.2f}%[/]")
        table.add_row("Month Return", f"[{'green' if month_return >= 0 else 'red'}]{month_return:+.2f}%[/]")
        table.add_row("YTD Return", f"[{'green' if ytd_return >= 0 else 'red'}]{ytd_return:+.2f}%[/]")
        table.add_row("Avg Volume (20d)", f"{avg_volume:,.0f}")
        table.add_row("52W High", f"Rp {df['high'].max():,.0f}")
        table.add_row("52W Low", f"Rp {df['low'].min():,.0f}")
        
        console.print(table)
    
    def _display_technical_analysis(self, df: pd.DataFrame):
        """Display technical indicators."""
        console.print("\n[bold]Technical Analysis[/bold]")
        
        # Calculate indicators
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
        rs = gain / loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
        
        latest = df.iloc[-1]
        
        table = Table(box=box.ROUNDED)
        table.add_column("Indicator", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Signal", justify="center")
        
        # RSI
        rsi = latest['rsi']
        rsi_signal = "[green]Oversold[/green]" if rsi < 30 else "[red]Overbought[/red]" if rsi > 70 else "[yellow]Neutral[/yellow]"
        table.add_row("RSI (14)", f"{rsi:.1f}", rsi_signal)
        
        # MACD
        macd = latest['macd']
        macd_sig = latest['macd_signal']
        macd_signal = "[green]Bullish[/green]" if macd > macd_sig else "[red]Bearish[/red]"
        table.add_row("MACD", f"{macd:.2f}", macd_signal)
        
        # Price vs MA
        price = latest['close']
        ma20 = latest['sma_20']
        ma50 = latest['sma_50']
        ma_signal = "[green]Above MAs[/green]" if price > ma20 and price > ma50 else "[red]Below MAs[/red]" if price < ma20 and price < ma50 else "[yellow]Mixed[/yellow]"
        table.add_row("Price vs MAs", f"vs MA20: {(price/ma20-1)*100:+.1f}%", ma_signal)
        
        # Volatility
        vol = latest['volatility']
        vol_signal = "[red]High[/red]" if vol > 40 else "[yellow]Medium[/yellow]" if vol > 20 else "[green]Low[/green]"
        table.add_row("Volatility (Ann.)", f"{vol:.1f}%", vol_signal)
        
        console.print(table)
    
    def _get_ml_prediction(self, df: pd.DataFrame) -> dict:
        """Get ML model prediction."""
        console.print("\n[bold]ML Prediction (XGBoost)[/bold]")
        
        if self.model is None:
            console.print("[yellow]No trained model found. Please train first.[/yellow]")
            return {'probability': 0.5, 'direction': 'NEUTRAL', 'confidence': 0}
        
        try:
            # Add features
            df_feat = df.copy()
            X = self.feature_engineer.prepare_inference_features(df_feat)
            
            if X.empty:
                console.print("[yellow]Could not prepare features.[/yellow]")
                return {'probability': 0.5, 'direction': 'NEUTRAL', 'confidence': 0}
            
            # Get prediction for latest row
            latest_features = X.iloc[-1:].values
            prob = self.model.model.predict_proba(latest_features)[0, 1]
            
            direction = "BULLISH" if prob > 0.55 else "BEARISH" if prob < 0.45 else "NEUTRAL"
            confidence = abs(prob - 0.5) * 200  # 0-100 scale
            
            color = "green" if direction == "BULLISH" else "red" if direction == "BEARISH" else "yellow"
            
            table = Table(box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            
            table.add_row("Up Probability", f"{prob*100:.1f}%")
            table.add_row("Direction", f"[{color}]{direction}[/{color}]")
            table.add_row("Confidence", f"{confidence:.0f}%")
            
            console.print(table)
            
            return {'probability': prob, 'direction': direction, 'confidence': confidence}
            
        except Exception as e:
            console.print(f"[red]Prediction failed: {e}[/red]")
            return {'probability': 0.5, 'direction': 'NEUTRAL', 'confidence': 0}
    
    def _analyze_signals(self, df: pd.DataFrame) -> dict:
        """Analyze all trading signals."""
        console.print("\n[bold]Signal Analysis[/bold]")
        
        try:
            signals_df = self.signal_combiner.calculate_signals(df)
            latest = signals_df.iloc[-1]
            
            table = Table(box=box.ROUNDED)
            table.add_column("Signal Type", style="cyan")
            table.add_column("Value", justify="right")
            table.add_column("Interpretation", justify="center")
            
            # Composite score
            score = latest.get('composite_score', 0)
            score_interp = "[green]Strong Buy[/green]" if score > 0.5 else "[green]Buy[/green]" if score > 0.2 else "[red]Sell[/red]" if score < -0.2 else "[yellow]Hold[/yellow]"
            table.add_row("Composite Score", f"{score:.2f}", score_interp)
            
            # Signal
            signal = latest.get('signal', 'HOLD')
            sig_color = "green" if signal == "BUY" else "red" if signal == "SELL" else "yellow"
            table.add_row("Signal", f"[{sig_color}]{signal}[/{sig_color}]", "")
            
            console.print(table)
            
            return {
                'composite_score': score,
                'signal': signal,
                'buy_signals': 3 if score > 0.3 else 2 if score > 0 else 1,
                'sell_signals': 3 if score < -0.3 else 2 if score < 0 else 1
            }
            
        except Exception as e:
            console.print(f"[yellow]Signal analysis failed: {e}[/yellow]")
            return {'composite_score': 0, 'signal': 'HOLD', 'buy_signals': 0, 'sell_signals': 0}
    
    def _get_market_regime(self) -> MarketRegime:
        """Get current market regime from IHSG index."""
        console.print("\n[bold]Market Regime[/bold]")
        
        try:
            # Fetch IHSG index using yfinance directly
            import yfinance as yf
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            ihsg = yf.Ticker("^JKSE")
            ihsg_data = ihsg.history(start=start_date, end=end_date, auto_adjust=True)
            
            if not ihsg_data.empty:
                ihsg_data = ihsg_data.rename(columns={'Close': 'close'})
                prices = ihsg_data['close']
                regime = self.regime_detector.detect_regime(prices)
                
                color = "red" if regime == MarketRegime.HIGH_VOL else "green" if regime == MarketRegime.LOW_VOL else "yellow"
                multiplier = self.regime_detector.get_position_multiplier(regime)
                
                console.print(f"  Current Regime: [{color}]{regime.value}[/{color}]")
                console.print(f"  Position Multiplier: {multiplier:.1f}x")
                
                return regime
        except Exception as e:
            console.print(f"[yellow]Could not determine regime: {e}[/yellow]")
        
        return MarketRegime.NORMAL
    
    def _display_historical_performance(self, df: pd.DataFrame):
        """Display historical performance statistics."""
        console.print("\n[bold]Historical Performance[/bold]")
        
        returns = df['close'].pct_change().dropna()
        
        # Monthly returns
        df_monthly = df.copy()
        df_monthly['date'] = pd.to_datetime(df_monthly['date'])
        df_monthly = df_monthly.set_index('date')
        monthly_returns = df_monthly['close'].resample('ME').last().pct_change().dropna()
        
        table = Table(box=box.ROUNDED)
        table.add_column("Period", style="cyan")
        table.add_column("Return", justify="right")
        table.add_column("Win Rate", justify="right")
        
        # Last 3 months
        for i, month_ret in enumerate(monthly_returns.tail(3)):
            month_name = monthly_returns.tail(3).index[i].strftime('%b %Y')
            color = "green" if month_ret >= 0 else "red"
            table.add_row(month_name, f"[{color}]{month_ret*100:+.1f}%[/{color}]", "")
        
        # Overall stats
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        daily_win_rate = positive_days / total_days * 100 if total_days > 0 else 0
        
        table.add_row("", "", "")
        table.add_row("Daily Win Rate", "", f"{daily_win_rate:.1f}%")
        table.add_row("Avg Daily Return", f"{returns.mean()*100:+.3f}%", "")
        table.add_row("Max Daily Gain", f"[green]{returns.max()*100:+.1f}%[/green]", "")
        table.add_row("Max Daily Loss", f"[red]{returns.min()*100:+.1f}%[/red]", "")
        
        console.print(table)
    
    def _generate_recommendation(
        self, 
        ticker: str, 
        df: pd.DataFrame, 
        prediction: dict, 
        signals: dict, 
        regime: MarketRegime,
        portfolio_value: float
    ):
        """Generate final recommendation."""
        console.print("\n" + "=" * 60)
        console.print(f"[bold][{ticker}] RECOMMENDATION[/bold]")
        console.print("=" * 60)
        
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
        bullish_factors = 0
        bearish_factors = 0
        
        if prediction['direction'] == 'BULLISH':
            bullish_factors += 2
        elif prediction['direction'] == 'BEARISH':
            bearish_factors += 2
        
        if signals['signal'] == 'BUY':
            bullish_factors += 2
        elif signals['signal'] == 'SELL':
            bearish_factors += 2
        
        if signals['composite_score'] > 0.2:
            bullish_factors += 1
        elif signals['composite_score'] < -0.2:
            bearish_factors += 1
        
        # Get volatility for sizing
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Position sizing
        position = self.position_sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            stock_volatility=volatility,
            avg_market_volatility=0.20,
            entry_price=entry_price,
            stop_loss=stop_loss,
            confidence=prediction['confidence'] / 100,
            market_regime=regime.value
        )
        
        # Recommendation
        if bullish_factors >= 4:
            action = "STRONG BUY"
            action_color = "bold green"
        elif bullish_factors >= 2 and bearish_factors < 2:
            action = "BUY"
            action_color = "green"
        elif bearish_factors >= 4:
            action = "STRONG SELL"
            action_color = "bold red"
        elif bearish_factors >= 2 and bullish_factors < 2:
            action = "SELL"
            action_color = "red"
        else:
            action = "HOLD"
            action_color = "yellow"
        
        console.print(f"\n[{action_color}]>>> {action} <<<[/{action_color}]\n")
        
        table = Table(box=box.ROUNDED, title="Position Details")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Entry Price", f"Rp {entry_price:,.0f}")
        table.add_row("Stop Loss", f"Rp {stop_loss:,.0f} ({(stop_loss/entry_price-1)*100:+.1f}%)")
        table.add_row("Take Profit", f"Rp {take_profit:,.0f} ({(take_profit/entry_price-1)*100:+.1f}%)")
        table.add_row("", "")
        table.add_row("Suggested Shares", f"{position['shares']:,}")
        table.add_row("Position Value", f"Rp {position['position_value']:,.0f}")
        table.add_row("Portfolio %", f"{position['position_pct']:.1f}%")
        table.add_row("Risk Amount", f"Rp {position['risk_amount']:,.0f}")
        table.add_row("", "")
        table.add_row("Bullish Factors", f"{bullish_factors}")
        table.add_row("Bearish Factors", f"{bearish_factors}")
        table.add_row("Regime Adjustment", f"{position['regime_adjustment']:.1f}x")
        
        console.print(table)
        
        # Disclaimer
        console.print("\n[dim]!! This is not financial advice. Always do your own research !![/dim]")


def main():
    parser = argparse.ArgumentParser(description='Single Stock Analysis')
    parser.add_argument('ticker', type=str, help='Stock ticker (e.g., BBCA.JK)')
    parser.add_argument('--portfolio', type=float, default=100_000_000, 
                        help='Portfolio value for position sizing (default: 100M IDR)')
    
    args = parser.parse_args()
    
    analyzer = StockAnalyzer()
    analyzer.analyze(args.ticker, args.portfolio)


if __name__ == "__main__":
    main()
