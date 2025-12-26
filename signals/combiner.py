"""
Signal Combiner Module
Combines multiple signal sources into a unified composite score
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from .technical import TechnicalIndicators
from .statistical import StatisticalSignals

logger = logging.getLogger(__name__)


class SignalCombiner:
    """
    Combines technical, statistical, and ML signals into a unified score.
    Implements the Jim Simons approach of combining multiple uncorrelated factors.
    """
    
    # Optimized weights: ML prediction is the primary driver
    DEFAULT_WEIGHTS = {
        'momentum': 0.10,
        'mean_reversion': 0.10,
        'volatility': 0.10,
        'technical': 0.10,
        'ml_prediction': 0.60
    }
    
    def __init__(
        self, 
        config=None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize signal combiner.
        
        Args:
            config: Configuration object
            weights: Custom signal weights (must sum to 1.0)
        """
        self.config = config
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Initialize sub-modules
        signal_config = config.signal if config else None
        self.technical = TechnicalIndicators(signal_config)
        self.statistical = StatisticalSignals(signal_config)
        
        # Validate weights sum to 1
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum}, normalizing...")
            for k in self.weights:
                self.weights[k] /= weight_sum
    
    def calculate_signals(
        self, 
        df: pd.DataFrame,
        ml_predictions: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate all signals and combine into composite score.
        
        Args:
            df: DataFrame with OHLCV data for a single ticker
            ml_predictions: Optional ML model predictions (0-1 probability)
            
        Returns:
            DataFrame with all signals and composite score
        """
        df = df.copy()
        
        # Technical indicators
        df = self.technical.calculate_all(df)
        
        # Statistical signals
        df = self.statistical.calculate_all(df)
        
        # Calculate individual scores for each row
        scores = []
        for idx, row in df.iterrows():
            score = self._calculate_composite_score(row, ml_predictions)
            scores.append(score)
        
        df['composite_score'] = scores
        
        # Rank scores (higher is better)
        df['score_rank'] = df['composite_score'].rank(pct=True)
        
        # Buy/Sell signals (Raised back to 0.3 for conservative strategy)
        df['signal'] = 'HOLD'
        df.loc[df['composite_score'] > 0.3, 'signal'] = 'BUY'
        df.loc[df['composite_score'] < -0.3, 'signal'] = 'SELL'
        
        return df
    
    def _calculate_composite_score(
        self, 
        row: pd.Series,
        ml_predictions: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate composite score for a single row.
        
        Combines multiple signals using weighted average.
        """
        score = 0.0
        
        # Momentum score
        momentum_score = self.statistical.get_momentum_score(row)
        score += self.weights['momentum'] * momentum_score
        
        # Mean reversion score
        mean_rev_score = self.statistical.get_mean_reversion_score(row)
        score += self.weights['mean_reversion'] * mean_rev_score
        
        # Volatility score
        vol_score = self.statistical.get_volatility_score(row)
        score += self.weights['volatility'] * vol_score
        
        # Technical score
        tech_score = self.technical.get_indicator_score(row)
        score += self.weights['technical'] * tech_score
        
        # ML prediction score (if available)
        if ml_predictions is not None and row.name in ml_predictions.index:
            ml_score = (ml_predictions.loc[row.name] - 0.5) * 2  # Convert 0-1 to -1 to 1
            score += self.weights['ml_prediction'] * ml_score
        
        return np.clip(score, -1, 1)
    
    def rank_stocks(
        self, 
        data: Dict[str, pd.DataFrame],
        ml_predictions: Optional[Dict[str, pd.Series]] = None
    ) -> pd.DataFrame:
        """
        Calculate signals for all stocks and rank them.
        
        Args:
            data: Dictionary mapping ticker to DataFrame
            ml_predictions: Optional ML predictions per ticker
            
        Returns:
            DataFrame with latest signals for each stock, ranked by score
        """
        rankings = []
        
        for ticker, df in data.items():
            if len(df) < 50:  # Need minimum data
                continue
            
            try:
                # Get ML predictions for this ticker if available
                ml_pred = ml_predictions.get(ticker) if ml_predictions else None
                
                # Calculate signals
                signals_df = self.calculate_signals(df, ml_pred)
                
                # Get latest row
                latest = signals_df.iloc[-1].copy()
                latest['ticker'] = ticker
                latest['date'] = df['date'].iloc[-1]
                
                rankings.append(latest)
                
            except Exception as e:
                logger.warning(f"Error calculating signals for {ticker}: {e}")
        
        if not rankings:
            return pd.DataFrame()
        
        # Combine and sort by score
        result = pd.DataFrame(rankings)
        result = result.sort_values('composite_score', ascending=False)
        result = result.reset_index(drop=True)
        
        return result
    
    def get_top_n(
        self, 
        rankings: pd.DataFrame, 
        n: int = 10,
        signal_filter: str = 'BUY'
    ) -> pd.DataFrame:
        """
        Get top N stocks from rankings.
        
        Args:
            rankings: DataFrame from rank_stocks()
            n: Number of stocks to return
            signal_filter: Only include stocks with this signal ('BUY', 'SELL', or None for all)
            
        Returns:
            Top N stocks
        """
        df = rankings.copy()
        
        if signal_filter:
            df = df[df['signal'] == signal_filter]
        
        return df.head(n)
    
    def calculate_batch(
        self, 
        all_data: pd.DataFrame,
        ml_predictions: Optional[Dict[str, pd.Series]] = None
    ) -> pd.DataFrame:
        """
        Calculate signals for batch data (multi-ticker DataFrame).
        
        Args:
            all_data: DataFrame with 'ticker' column and multiple stocks
            ml_predictions: Optional ML predictions
            
        Returns:
            DataFrame with signals for all stocks
        """
        results = []
        
        for ticker in all_data['ticker'].unique():
            ticker_data = all_data[all_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')
            
            if len(ticker_data) < 50:
                continue
            
            ml_pred = ml_predictions.get(ticker) if ml_predictions else None
            signals = self.calculate_signals(ticker_data, ml_pred)
            results.append(signals)
        
        if not results:
            return pd.DataFrame()
        
        return pd.concat(results, ignore_index=True)
