"""
Signal Combiner Module
Simplified for Deep Learning Pipeline.
Primarily passes through ML confidence scores.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class SignalCombiner:
    """
    Combines signals. In the LSTM architecture, this primarily
    validates and formats the Deep Learning model output.
    """
    
    def __init__(self, config=None):
        self.config = config
        
    def calculate_signals(self, df: pd.DataFrame, ml_predictions: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Derive final signals.
        """
        df = df.copy()
        
        # Default to 0 score
        df['composite_score'] = 0.0
        df['signal'] = 'HOLD'
        
        # If ML predictions provided, use them directly
        if ml_predictions is not None:
            # Align indices
            common_indices = df.index.intersection(ml_predictions.index)
            if not common_indices.empty:
                # ml_predictions are expected to be 0-1 probabilities (or confidence)
                # We map this to -1 to 1 score
                # Assuming ML prediction is Probability of Class 2 (Profit)
                
                probs = ml_predictions.loc[common_indices]
                # Score: (Prob - 0.33) * Scale? 
                # Or just use the Prob directly for ranking.
                # Let's say Score = Prob of Profit
                
                df.loc[common_indices, 'composite_score'] = probs
                
                # Signal Generation
                # Simple threshold: > 50% prob? Or > relative.
                # Let's use 0.5 as base, but this should be tuned.
                df.loc[df['composite_score'] > 0.5, 'signal'] = 'BUY'
                
        return df

    def rank_stocks(self, data: Dict[str, pd.DataFrame], ml_predictions: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """
        Rank stocks based on ML scores.
        """
        rankings = []
        for ticker, df in data.items():
            if len(df) < 20: continue
            
            ml_pred = ml_predictions.get(ticker) if ml_predictions else None
            
            if ml_pred is not None and not ml_pred.empty:
                latest_score = ml_pred.iloc[-1]
                latest_price = df['close'].iloc[-1]
                
                rankings.append({
                    'ticker': ticker,
                    'price': latest_price,
                    'composite_score': latest_score,
                    'signal': 'BUY' if latest_score > 0.5 else 'HOLD',
                    'date': df['date'].iloc[-1]
                })
                
        if not rankings:
            return pd.DataFrame()
            
        return pd.DataFrame(rankings).sort_values('composite_score', ascending=False)
