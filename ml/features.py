"""
Feature Engineering Module
Implements Raw OHLCV Sequence Generation and Triple Barrier Labeling
Based on: "Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data"
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class TBLDataset(Dataset):
    """PyTorch Dataset for TBL/LSTM training."""
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class FeatureEngineer:
    """
    Handles data preprocessing:
    1. Triple Barrier Labeling
    2. Raw OHLCV Sequence Generation
    3. Z-Score Normalization
    """
    
    def __init__(self, config):
        self.window_size = config.data.window_size
        self.horizon = config.ml.tbl_horizon
        self.barrier = config.ml.tbl_barrier
        
    def create_sequences(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create (X, y) sequences from DataFrame.
        
        Args:
            df: DataFrame with date, open, high, low, close, volume columns.
            is_training: If True, generates labels.
            
        Returns:
            X: (N, Window, 5) array
            y: (N,) array of labels (if training) or None
        """
        # Ensure date sorted
        df = df.sort_values('date').reset_index(drop=True)
        
        # Extract raw arrays
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        
        # Combine into (L, 5)
        # Note: Order matters -> Open, High, Low, Close, Volume
        data = np.stack([opens, highs, lows, closes, volumes], axis=1)
        
        sequences = []
        labels = []
        
        # Indices valid for sequence generation
        # We need 'window_size' history
        start_idx = self.window_size
        
        # For training, we need 'horizon' future for labeling
        end_idx = len(df) - self.horizon if is_training else len(df)
        
        for i in range(start_idx, end_idx):
            # 1. Extract Window (lookback)
            # Shapes: (Window, 5)
            window_data = data[i-self.window_size : i]
            
            # 2. Normalize Window (Z-Score)
            # Normalize each feature independently within the window to handle non-stationarity
            mean = np.mean(window_data, axis=0)
            std = np.std(window_data, axis=0)
            # Avoid division by zero (e.g. constant price/volume in window)
            std[std == 0] = 1.0 
            
            normalized_window = (window_data - mean) / std
            
            if is_training:
                # 3. Generate Label (Triple Barrier)
                current_close = closes[i-1] # The price at end of window (t-1) which is "current time" for prediction
                # Actually, if we predict for time t, we use info up to t-1.
                # Paper: "prediction horizon of 29 days".
                # Standard: Input [t-W : t], Predict [t : t+H] or Label based on [t : t+H] relative to Close[t].
                # Let's align: i is current time index. Window is [i-W : i]. Close[i-1] is the latest known price.
                # Future window starts at i.
                
                # Correction: Window ends at i (exclusive in slice), so last element is i-1.
                # Entry price is 'close' at i-1.
                entry_price = closes[i-1]
                
                future_highs = highs[i : i + self.horizon]
                future_lows = lows[i : i + self.horizon]
                
                label = self._get_label(entry_price, future_highs, future_lows)
                labels.append(label)
            
            sequences.append(normalized_window)
            
        if not sequences:
            return np.array([]), np.array([]) if is_training else None
            
        X = np.array(sequences)
        y = np.array(labels) if is_training else None
        
        return X, y
    
    def _get_label(self, entry_price: float, future_highs: np.ndarray, future_lows: np.ndarray) -> int:
        """
        Compute Triple Barrier Label.
        0: Stop Loss Hit
        1: Time Limit / No Move / Double Hit
        2: Take Profit Hit
        """
        tp_price = entry_price * (1 + self.barrier)
        sl_price = entry_price * (1 - self.barrier)
        
        # Identify hit indices
        tp_hits = np.where(future_highs >= tp_price)[0]
        sl_hits = np.where(future_lows <= sl_price)[0]
        
        first_tp = tp_hits[0] if len(tp_hits) > 0 else 99999
        first_sl = sl_hits[0] if len(sl_hits) > 0 else 99999
        
        if first_tp == 99999 and first_sl == 99999:
            return 1 # Time Limit
        
        if first_tp < first_sl:
            return 2 # Profit
        elif first_sl < first_tp:
            return 0 # Loss
        else:
            # Same day touch
            return 1 # Paper rule: "If both... hit... on the same day -> Time Limit"

    def prepare_inference(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare latest sequence for inference from DataFrame."""
        # Creates a SINGLE sequence using the last 'window_size' rows
        if len(df) < self.window_size:
            raise ValueError(f"Insufficient data: {len(df)} < {self.window_size}")
            
        df = df.sort_values('date')
        
        opens = df['open'].values[-self.window_size:]
        highs = df['high'].values[-self.window_size:]
        lows = df['low'].values[-self.window_size:]
        closes = df['close'].values[-self.window_size:]
        volumes = df['volume'].values[-self.window_size:]
        
        window_data = np.stack([opens, highs, lows, closes, volumes], axis=1)
        
        # Normalize
        mean = np.mean(window_data, axis=0)
        std = np.std(window_data, axis=0)
        std[std == 0] = 1.0
        norm_window = (window_data - mean) / std
        
        # Add batch dim: (1, W, 5)
        tensor = torch.FloatTensor(norm_window).unsqueeze(0)
        return tensor
