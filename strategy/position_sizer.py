"""
Position Sizing Module
Kelly-criterion inspired position sizing with volatility adjustment
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculates position sizes based on volatility, risk, and portfolio constraints.
    Uses Kelly-criterion inspired approach with conservative fraction.
    """
    
    def __init__(self, config=None):
        """
        Initialize position sizer.
        
        Args:
            config: PortfolioConfig object (optional)
        """
        if config:
            self.max_positions = config.max_positions
            self.base_position_pct = config.base_position_pct
            self.max_sector_exposure = config.max_sector_exposure
            self.max_portfolio_vol = config.max_portfolio_volatility
            self.max_correlation = config.max_correlation
        else:
            self.max_positions = 10
            self.base_position_pct = 0.10
            self.max_sector_exposure = 0.30
            self.max_portfolio_vol = 0.25
            self.max_correlation = 0.7
    
    def calculate_atr_based_size(
        self,
        portfolio_value: float,
        entry_price: float,
        atr: float,
        risk_pct: float = 0.01,
        atr_multiplier: float = 2.0
    ) -> Dict:
        """
        Calculate position size using ATR-based volatility targeting.

        Hypothesis 3: Volatility Targeting
        Formula: Shares = Risk Budget / (ATR * Multiplier)

        Args:
            portfolio_value: Total portfolio value
            entry_price: Entry price per share
            atr: Average True Range (volatility measure)
            risk_pct: Portfolio risk per trade (default 1%)
            atr_multiplier: ATR multiplier for risk distance (default 2.0)

        Returns:
            Dictionary with position sizing details
        """
        if atr <= 0 or entry_price <= 0:
            return {
                'shares': 0,
                'position_value': 0.0,
                'position_pct': 0.0,
                'risk_amount': 0.0,
                'risk_pct': 0.0,
                'sizing_method': 'atr_volatility_targeting'
            }

        # Risk budget for this trade
        risk_budget = portfolio_value * risk_pct

        # Risk per share = ATR * Multiplier
        risk_per_share = atr * atr_multiplier

        # Calculate shares: Risk Budget / Risk Per Share
        shares = int(risk_budget / risk_per_share)

        # Actual position value
        position_value = shares * entry_price

        # Actual risk (if hit stop loss)
        actual_risk = shares * risk_per_share

        return {
            'shares': shares,
            'position_value': round(position_value, 2),
            'position_pct': round(position_value / portfolio_value * 100, 2),
            'risk_amount': round(actual_risk, 2),
            'risk_pct': round(actual_risk / portfolio_value * 100, 2),
            'atr': round(atr, 2),
            'atr_multiplier': atr_multiplier,
            'risk_per_share': round(risk_per_share, 2),
            'sizing_method': 'atr_volatility_targeting'
        }

    def calculate_position_size(
        self,
        portfolio_value: float,
        stock_volatility: float,
        avg_market_volatility: float,
        entry_price: float,
        stop_loss: float,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.5,
        confidence: float = 0.5,
        market_regime: Optional[Literal["HIGH_VOL", "LOW_VOL", "NORMAL"]] = None,
        atr: Optional[float] = None,
        use_atr_targeting: bool = True
    ) -> Dict:
        """
        Calculate position size using multiple factors.

        Args:
            portfolio_value: Total portfolio value
            stock_volatility: Stock's annualized volatility
            avg_market_volatility: Average market volatility
            entry_price: Planned entry price
            stop_loss: Stop loss price
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average winner / average loser
            confidence: Signal confidence (0-1)
            market_regime: Market regime (HIGH_VOL, LOW_VOL, NORMAL)
            atr: Average True Range (if available)
            use_atr_targeting: If True and ATR available, use ATR-based sizing as primary method

        Returns:
            Dictionary with position sizing details
        """
        # HYPOTHESIS 3: ATR-based Volatility Targeting (Primary Method)
        if use_atr_targeting and atr is not None and atr > 0:
            size_info = self.calculate_atr_based_size(
                portfolio_value=portfolio_value,
                entry_price=entry_price,
                atr=atr,
                risk_pct=0.01,  # 1% risk per trade
                atr_multiplier=2.0
            )

            # Apply confidence and regime adjustments
            confidence_mult = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x
            regime_mult = self._get_regime_multiplier(market_regime)

            # Adjust position size
            adjusted_shares = int(size_info['shares'] * confidence_mult * regime_mult)
            size_info['shares'] = adjusted_shares
            size_info['position_value'] = round(adjusted_shares * entry_price, 2)
            size_info['position_pct'] = round(size_info['position_value'] / portfolio_value * 100, 2)
            size_info['confidence_mult'] = round(confidence_mult, 2)
            size_info['regime_mult'] = round(regime_mult, 2)

            # Apply max position constraint
            max_position = portfolio_value / self.max_positions
            if size_info['position_value'] > max_position:
                size_info['shares'] = int(max_position / entry_price)
                size_info['position_value'] = round(size_info['shares'] * entry_price, 2)
                size_info['position_pct'] = round(size_info['position_value'] / portfolio_value * 100, 2)

            return size_info

        # LEGACY METHOD: Multi-factor approach (fallback if ATR not available)
        # Base position size
        base_size = portfolio_value * self.base_position_pct
        
        # Volatility adjustment (inverse relationship)
        vol_ratio = avg_market_volatility / stock_volatility if stock_volatility > 0 else 1
        vol_adjustment = np.clip(vol_ratio, 0.5, 2.0)  # Limit adjustment range
        
        # Kelly criterion (half-Kelly for conservatism)
        kelly_pct = self._calculate_kelly(win_rate, avg_win_loss_ratio)
        kelly_size = portfolio_value * kelly_pct * 0.5  # Half-Kelly
        
        # Risk-based sizing (risk fixed amount per trade)
        risk_per_trade = portfolio_value * 0.01  # 1% risk per trade
        risk_per_share = abs(entry_price - stop_loss)
        shares_by_risk = risk_per_trade / risk_per_share if risk_per_share > 0 else 0
        risk_based_size = shares_by_risk * entry_price
        
        # Confidence adjustment
        confidence_mult = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x
        
        # Take minimum of all sizing methods (conservative)
        sizes = [base_size, kelly_size, risk_based_size]
        position_value = min(sizes) * vol_adjustment * confidence_mult
        
        # Apply maximum position constraint
        max_position = portfolio_value / self.max_positions
        position_value = min(position_value, max_position)
        
        # Apply market regime multiplier
        regime_mult = self._get_regime_multiplier(market_regime)
        position_value = position_value * regime_mult
        
        # Calculate shares
        shares = int(position_value / entry_price) if entry_price > 0 else 0
        actual_value = shares * entry_price
        
        return {
            'shares': shares,
            'position_value': round(actual_value, 2),
            'position_pct': round(actual_value / portfolio_value * 100, 2),
            'risk_amount': round(shares * risk_per_share, 2),
            'risk_pct': round(shares * risk_per_share / portfolio_value * 100, 2),
            'vol_adjustment': round(vol_adjustment, 2),
            'regime_adjustment': round(regime_mult, 2),
            'kelly_fraction': round(kelly_pct, 3),
            'sizing_method': 'risk_adjusted'
        }
    
    def _get_regime_multiplier(self, regime: Optional[str]) -> float:
        """
        Get position size multiplier based on market regime.

        Args:
            regime: Market regime string (CRASH, HIGH_VOL, LOW_VOL, NORMAL)

        Returns:
            Multiplier to apply to position size
        """
        if regime == "CRASH":
            return 0.0  # No trades during market crash
        elif regime == "HIGH_VOL":
            return 0.7  # Reduce exposure in high volatility
        elif regime == "LOW_VOL":
            return 1.2  # Slightly increase in calm markets
        else:
            return 1.0  # Default
    
    def _calculate_kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly fraction.
        
        Kelly % = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        """
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        # Ensure Kelly is between 0 and 0.25 (capped for safety)
        return np.clip(kelly, 0, 0.25)
    
    def allocate_portfolio(
        self,
        portfolio_value: float,
        candidates: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        sector_mapping: Dict[str, str],
        existing_positions: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Allocate portfolio across multiple candidates.
        
        Args:
            portfolio_value: Total portfolio value
            candidates: DataFrame with candidate stocks (must have 'ticker', 'composite_score')
            price_data: Dictionary with price history per ticker
            sector_mapping: Ticker to sector mapping
            existing_positions: Current position values
            
        Returns:
            DataFrame with allocation details
        """
        existing_positions = existing_positions or {}
        available_capital = portfolio_value - sum(existing_positions.values())
        
        allocations = []
        sector_exposure = {}
        
        for _, row in candidates.iterrows():
            ticker = row['ticker']
            
            if ticker in existing_positions:
                continue  # Skip already held positions
            
            # Get sector
            sector = sector_mapping.get(ticker, 'Other')
            
            # Check sector exposure
            current_sector_exposure = sector_exposure.get(sector, 0)
            if current_sector_exposure >= self.max_sector_exposure * portfolio_value:
                continue
            
            # Get volatility
            if ticker in price_data:
                returns = price_data[ticker]['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) if len(returns) > 20 else 0.3
            else:
                volatility = 0.3  # Default
            
            # Get price and ATR
            entry_price = row.get('close', row.get('entry_price', 0))
            atr = row.get('atr', entry_price * 0.02)
            stop_loss = entry_price - (2 * atr)
            
            # Calculate position size
            size_info = self.calculate_position_size(
                portfolio_value=available_capital,
                stock_volatility=volatility,
                avg_market_volatility=0.20,  # Assumed avg
                entry_price=entry_price,
                stop_loss=stop_loss,
                confidence=max(0.3, min(1.0, row.get('composite_score', 0.5) + 0.5))
            )
            
            # Check if we have enough capital
            if size_info['position_value'] > available_capital * 0.9:
                break
            
            allocation = {
                'ticker': ticker,
                'sector': sector,
                'score': row.get('composite_score', 0),
                'entry_price': entry_price,
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(entry_price + 3 * atr, 2),
                **size_info
            }
            
            allocations.append(allocation)
            
            # Update tracking
            available_capital -= size_info['position_value']
            sector_exposure[sector] = sector_exposure.get(sector, 0) + size_info['position_value']
            
            # Stop if max positions reached
            if len(allocations) + len(existing_positions) >= self.max_positions:
                break
        
        return pd.DataFrame(allocations)
    
    def calculate_correlation_penalty(
        self,
        returns: pd.DataFrame,
        new_ticker: str,
        existing_tickers: List[str]
    ) -> float:
        """
        Calculate position size penalty based on correlation with existing positions.
        
        Returns multiplier between 0.5 and 1.0
        """
        if not existing_tickers or new_ticker not in returns.columns:
            return 1.0
        
        new_returns = returns[new_ticker]
        max_corr = 0
        
        for existing in existing_tickers:
            if existing in returns.columns:
                corr = new_returns.corr(returns[existing])
                max_corr = max(max_corr, abs(corr))
        
        # Reduce position if high correlation
        if max_corr > self.max_correlation:
            return 0.5
        elif max_corr > 0.5:
            return 0.75
        
        return 1.0
    
    def rebalance_positions(
        self,
        portfolio_value: float,
        current_positions: Dict[str, Dict],
        target_allocations: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Generate rebalancing orders.
        
        Returns dictionary of orders to execute.
        """
        orders = {}
        
        target_tickers = set(target_allocations['ticker'].tolist())
        current_tickers = set(current_positions.keys())
        
        # Positions to close
        for ticker in current_tickers - target_tickers:
            orders[ticker] = {
                'action': 'CLOSE',
                'shares': current_positions[ticker].get('shares', 0)
            }
        
        # Positions to open or adjust
        for _, row in target_allocations.iterrows():
            ticker = row['ticker']
            target_shares = row['shares']
            
            if ticker in current_positions:
                current_shares = current_positions[ticker].get('shares', 0)
                if target_shares > current_shares:
                    orders[ticker] = {
                        'action': 'ADD',
                        'shares': target_shares - current_shares,
                        'entry_price': row['entry_price']
                    }
                elif target_shares < current_shares:
                    orders[ticker] = {
                        'action': 'REDUCE',
                        'shares': current_shares - target_shares
                    }
            else:
                orders[ticker] = {
                    'action': 'OPEN',
                    'shares': target_shares,
                    'entry_price': row['entry_price'],
                    'stop_loss': row['stop_loss'],
                    'take_profit': row['take_profit']
                }
        
        return orders
