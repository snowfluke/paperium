"""
Feature Engineering Module
Creates ML features from price data and signals
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging

from signals.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates features for machine learning models.
    Generates price-based, technical, and statistical features.
    """
    
    # EXACT 46-feature set expected by the champion model
    LEGACY_46_FEATURES = [
        'return_1d', 'return_5d', 'return_20d', 'sma_20', 'sma_50', 'rsi',
        'macd', 'macd_signal', 'macd_hist', 'atr', 'volatility', 'volume_sma',
        'rel_volume', 'return_2d', 'return_3d', 'return_10d', 'log_return',
        'volatility_5d', 'volatility_20d', 'vol_ratio', 'price_to_ma10',
        'price_to_ma20', 'price_to_ma50', 'hl_range', 'hl_range_avg', 'gap',
        'close_position', 'volume_change', 'volume_ma20', 'relative_volume',
        'volatility_zscore', 'intraday_range_pct', 'mean_reversion_strength',
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
        'rsi_lag1', 'rsi_change', 'is_month_start', 'is_month_end',
        'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4'
    ]
    
    def __init__(self, config=None):
        """
        Initialize feature engineer.
        
        Args:
            config: MLConfig object (optional)
        """
        if config:
            self.feature_lags = config.feature_lags
            self.target_horizon = getattr(config, 'target_horizon', 5)  # Default 5 for day trading
        else:
            self.feature_lags = [1, 2, 3, 5, 10, 20]
            self.target_horizon = 5  # 5-day forward prediction for day trading
            
        # Initialize sub-modules for internal indicator calculation
        self.technical = TechnicalIndicators()
    
    def create_features(
        self, 
        df: pd.DataFrame,
        target_horizon: int = 1,
        include_raw_return: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Create feature matrix and target variable.
        
        Args:
            df: DataFrame with OHLCV and indicator columns
            target_horizon: Days ahead to predict (default 1 = next day)
            include_raw_return: Whether to return the raw float return for weighting
            
        Returns:
            Tuple of (feature DataFrame, target Series, optional raw_return Series)
        """
        df = df.copy()
        
        # Create target: next-day return
        df['target'] = df['close'].shift(-target_horizon) / df['close'] - 1
        df['target_direction'] = (df['target'] > 0).astype(int)
        
        # 1. Base technical indicators (SMA, RSI, MACD, ATR)
        df = self._add_base_indicators(df)
        
        # 2. Price-based features (Volatility, Returns, MA Relatives)
        df = self._add_price_features(df)
        
        # 3. Lagged features (Returns, Indicators)
        df = self._add_lagged_features(df)
        
        # 4. Calendar features (Day of week, Month start/end)
        df = self._add_calendar_features(df)
        
        # 5. Drop rows with NaN
        df = df.dropna()
        
        # 6. Ensure EXACT feature list and order
        X = df[self.LEGACY_46_FEATURES]
        y = df['target_direction']
        
        if include_raw_return:
            return X, y, df['target']

        return X, y, None
    
    
    def _add_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add base technical indicators expected by the 46-feature champion model."""
        # SMA
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        df = self.technical.add_rsi(df)
        
        # MACD (Mapping to legacy names: macd_hist)
        df = self.technical.add_macd(df)
        df['macd_hist'] = df['macd_histogram']
        
        # ATR
        df = self.technical.add_atr(df)
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['rel_volume'] = df['volume'] / df['volume_sma'].replace(0, 1)
        
        # Legacy Volatility (std of log returns * sqrt(252))
        if 'log_return' not in df.columns:
            # Use small epsilon to avoid log(0) errors
            ratio = (df['close'] / df['close'].shift(1)).replace([0, np.inf, -np.inf], np.nan).fillna(1.0)
            df['log_return'] = np.log(ratio.abs())

        df['volatility'] = df['log_return'].rolling(20).std() * np.sqrt(252)
        
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns at various horizons
        for lag in self.feature_lags:
            if f'return_{lag}d' not in df.columns:
                df[f'return_{lag}d'] = df['close'].pct_change(lag)
        
        # Ensure log_return exists
        if 'log_return' not in df.columns:
            ratio = (df['close'] / df['close'].shift(1)).replace([0, np.inf, -np.inf], np.nan).fillna(1.0)
            df['log_return'] = np.log(ratio.abs())


        
        # Volatility
        df['volatility_5d'] = df['log_return'].rolling(5).std() * np.sqrt(252)
        df['volatility_20d'] = df['log_return'].rolling(20).std() * np.sqrt(252)
        df['vol_ratio'] = df['volatility_5d'] / df['volatility_20d'].replace(0, np.inf)
        
        # Price relative to moving averages
        for period in [10, 20, 50]:
            ma = df['close'].rolling(period).mean()
            df[f'price_to_ma{period}'] = df['close'] / ma - 1
        
        # High-Low range (normalized)
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_avg'] = df['hl_range'].rolling(20).mean()
        
        # Gap (overnight return proxy)
        df['gap'] = df['open'] / df['close'].shift(1) - 1
        
        # Close position within day range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma20'].replace(0, 1)
        
        # --- NEW QUANT FEATURES ---
        
        # Volatility Z-Score: Detects abnormal volatility periods
        vol_mean = df['volatility_20d'].rolling(60).mean()
        vol_std = df['volatility_20d'].rolling(60).std()
        df['volatility_zscore'] = (df['volatility_20d'] - vol_mean) / vol_std.replace(0, 1)
        
        # Intraday Range Percentile: Where the close is within the daily range (averaged)
        df['intraday_range_pct'] = df['close_position'].rolling(5).mean()
        
        # Mean Reversion Strength: Standardized distance from 20-day MA
        price_to_ma_std = df['price_to_ma20'].rolling(60).std()
        df['mean_reversion_strength'] = df['price_to_ma20'].abs() / price_to_ma_std.replace(0, 1)
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged versions of key features."""
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            if 'return_1d' in df.columns:
                df[f'return_lag_{lag}'] = df['return_1d'].shift(lag)
        
        # Lagged indicators (if present)
        if 'rsi' in df.columns:
            df['rsi_lag1'] = df['rsi'].shift(1)
            df['rsi_change'] = df['rsi'] - df['rsi'].shift(1)
        
        # MACD/ZScore changes if present (not strictly in the 46 set but useful)
        if 'macd_histogram' in df.columns:
            df['macd_hist_lag1'] = df['macd_histogram'].shift(1)
            df['macd_hist_change'] = df['macd_histogram'] - df['macd_histogram'].shift(1)
        
        if 'zscore' in df.columns:
            df['zscore_lag1'] = df['zscore'].shift(1)
            df['zscore_change'] = df['zscore'] - df['zscore'].shift(1)
        
        return df
    
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features."""
        if 'date' in df.columns:
            # Convert to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Extract datetime components using apply to avoid type issues
            df['day_of_week'] = df['date'].apply(lambda x: x.dayofweek if pd.notna(x) else 0)
            df['month'] = df['date'].apply(lambda x: x.month if pd.notna(x) else 1)
            df['is_month_start'] = df['date'].apply(lambda x: int(x.is_month_start) if pd.notna(x) else 0)
            df['is_month_end'] = df['date'].apply(lambda x: int(x.is_month_end) if pd.notna(x) else 0)
            
            # One-hot encode day of week
            for day in range(5):  # Mon-Fri
                df[f'dow_{day}'] = (df['day_of_week'] == day).astype(int)
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names."""
        # Exclude non-feature columns
        exclude = {
            'date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
            'target', 'target_direction', 'signal', 'composite_score',
            'score_rank', 'day_of_week', 'month', 'created_at'
        }
        
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]
        
        return feature_cols
    
    def prepare_inference_features(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare features for inference (without target).
        
        Args:
            df: DataFrame with OHLCV and indicator columns
            
        Returns:
            Feature DataFrame ready for prediction
        """
        df = df.copy()
        
        df = self._add_base_indicators(df)
        df = self._add_price_features(df)
        df = self._add_lagged_features(df)
        df = self._add_calendar_features(df)
        
        X = df[self.LEGACY_46_FEATURES].fillna(0)
        
        return X
    
    def get_feature_importance(
        self, 
        model, 
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        return importance
