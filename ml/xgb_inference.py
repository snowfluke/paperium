#!/usr/bin/env python3
"""
XGBoost Inference Wrapper (Paperium V1 Integration)
Loads and uses V1's XGBoost model for enhanced signal generation.
"""
import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
import xgboost as xgb


class XGBoostInference:
    """Wrapper for V1 XGBoost model inference."""

    def __init__(self):
        self.model: Optional[Union[xgb.XGBClassifier, xgb.Booster]] = None
        self.metadata: Dict[str, float] = {}
        self.feature_names: List[str] = []
        self.is_loaded: bool = False

    def load_model(self, model_path="models/global_xgb_champion.pkl"):
        """Load V1 XGBoost model and metadata."""
        try:
            # V1 saves model in two parts:
            # 1. .pkl file with metadata (feature_names, config, etc.)
            # 2. .json file with actual XGBoost booster

            # Load metadata from pickle
            model_metadata = None
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_metadata = pickle.load(f)
                    # This is a dict with keys: feature_names, last_trained, performance_history, config

            # Load actual model from JSON
            json_path = model_path.replace('.pkl', '.json')
            if os.path.exists(json_path):
                # Create XGBClassifier wrapper
                self.model = xgb.XGBClassifier()
                # Load booster from JSON
                booster = xgb.Booster()
                booster.load_model(json_path)
                self.model._Booster = booster
            else:
                # Fallback: try legacy format where model is in pkl
                if model_metadata and 'model' in model_metadata and model_metadata['model'] is not None:
                    self.model = model_metadata['model']
                else:
                    return False

            # Store feature names for later use
            self.feature_names = model_metadata.get('feature_names', []) if model_metadata else []

            # Load champion metadata (performance metrics)
            metadata_path = "models/champion_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.metadata = data.get('xgboost', {})
            else:
                self.metadata = {}

            self.is_loaded = True
            return True

        except Exception as e:
            print(f"Failed to load XGBoost model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_features_v1(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate V1 complete feature set (56 features).
        UNIVERSAL FEATURE SET from paperium-v1 - NO simplification.

        Base 51 features + 5 intraday behavior proxies:
        - Technical indicators: RSI, MACD, ATR, SMA
        - Advanced volume: MFI, CMF, ADL, OBV trend, volume-price alignment
        - Price patterns: returns, volatility, gaps, shadows
        - Lagged features and calendar features
        """
        if len(df) < 60:
            return None

        data = df.copy()

        # Returns at various horizons
        for period in [1, 2, 3, 5, 10, 20]:
            data[f'return_{period}d'] = data['close'].pct_change(period)

        data['log_return'] = np.log(data['close'] / data['close'].shift(1))

        # Moving Averages
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()

        data['price_to_ma10'] = data['close'] / data['sma_10'] - 1
        data['price_to_ma20'] = data['close'] / data['sma_20'] - 1
        data['price_to_ma50'] = data['close'] / data['sma_50'] - 1

        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_lag1'] = data['rsi'].shift(1)
        data['rsi_change'] = data['rsi'] - data['rsi_lag1']

        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema12 - ema26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']

        # ATR (Critical for SL/TP calculation)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.DataFrame({
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        })
        true_range = ranges.max(axis=1)
        data['atr'] = true_range.rolling(14).mean()
        data['atr_pct'] = data['atr'] / data['close']

        # Volatility
        # Legacy volatility (std of log returns * sqrt(252))
        data['volatility'] = data['log_return'].rolling(20).std() * np.sqrt(252)
        data['volatility_5d'] = data['log_return'].rolling(5).std() * np.sqrt(252)
        data['volatility_20d'] = data['log_return'].rolling(20).std() * np.sqrt(252)
        data['vol_ratio'] = data['volatility_5d'] / (data['volatility_20d'] + 1e-8)
        data['vol_zscore'] = (data['volatility_5d'] - data['volatility_20d'].rolling(20).mean()) / (data['volatility_20d'].rolling(20).std() + 1e-8)

        # Basic Volume indicators
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['rel_volume'] = data['volume'] / (data['volume_sma'] + 1)
        data['volume_change'] = data['volume'].pct_change()
        data['volume_ma20'] = data['volume'].rolling(20).mean()
        data['relative_volume'] = data['volume'] / (data['volume_ma20'] + 1)

        # Advanced Volume Indicators (2024 research)
        # 1. Money Flow Index (MFI) - RSI with volume
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        mf_ratio = positive_mf / (negative_mf + 1e-8)
        data['mfi'] = 100 - (100 / (1 + mf_ratio))

        # 2. Chaikin Money Flow (CMF) - buying/selling pressure
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'] + 1e-8)
        clv = clv.fillna(0)
        mf_volume = clv * data['volume']
        data['cmf'] = (
            mf_volume.rolling(window=20).sum() /
            (data['volume'].rolling(window=20).sum() + 1e-8)
        )

        # 3. Accumulation/Distribution Line (ADL) - cumulative volume indicator
        adl_mf_volume = clv * data['volume']
        data['adl'] = adl_mf_volume.cumsum()
        data['adl_roc'] = data['adl'].pct_change(20)

        # 4. On-Balance Volume (OBV) trend
        obv = [0]
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.append(obv[-1] + data['volume'].iloc[i])
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.append(obv[-1] - data['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        data['obv'] = obv
        data['obv_trend'] = data['obv'].pct_change(10)

        # 5. Volume-Price Alignment - correlation between price and OBV movement
        data['volume_price_trend'] = data['close'].pct_change(10) * data['obv_trend']

        # Price patterns
        data['hl_range'] = (data['high'] - data['low']) / data['close']
        data['hl_range_avg'] = data['hl_range'].rolling(20).mean()
        data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
        data['intraday_range_pct'] = data['close_position'].rolling(5).mean()

        # Mean reversion strength (standardized distance from 20-day MA)
        data['mean_reversion_strength'] = (data['close'] - data['sma_20']) / (data['volatility_20d'] * data['close'] + 1e-8)

        # Intraday behavior proxies
        body = np.abs(data['close'] - data['open'])
        upper_shadow = data['high'] - data[['close', 'open']].max(axis=1)
        data['upper_shadow_ratio'] = upper_shadow / (body + 1e-8)
        data['gap_size'] = data['gap']
        data['gap_followthrough'] = (data['close'] - data['open']) / (data['open'] + 1e-8)
        data['intraday_momentum'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
        data['fade_signal'] = data['upper_shadow_ratio'] * np.abs(data['gap_size'])

        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            data[f'return_lag_{lag}'] = data['return_1d'].shift(lag)

        # Calendar features
        if 'date' in list(data.columns):
            date_col = pd.to_datetime(data['date'])
            data['date'] = date_col
            # Day of week (0=Monday, 4=Friday)
            dow_series = date_col.dt.day_of_week
            data['dow'] = dow_series
            for i in range(5):
                data[f'dow_{i}'] = (dow_series == i).astype(int)
            # Month start/end flags
            data['is_month_start'] = date_col.dt.is_month_start.astype(int)
            data['is_month_end'] = date_col.dt.is_month_end.astype(int)
        else:
            # Default to 0 if no date column
            for i in range(5):
                data[f'dow_{i}'] = 0
            data['is_month_start'] = 0
            data['is_month_end'] = 0

        return data

    def predict(self, df: pd.DataFrame) -> Optional[Dict[str, Union[float, str]]]:
        """
        Run XGBoost prediction on dataframe.
        Returns: Dict with confidence_score, sl_pct, tp_pct, etc. or None
        """
        if not self.is_loaded or self.model is None:
            return None

        # Calculate features
        features_df = self.calculate_features_v1(df)
        if features_df is None:
            return None

        # Get last row (most recent)
        latest_df = features_df.tail(1).copy()

        # Extract feature columns (exclude OHLCV and derived columns)
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker',
                       'sma_10', 'sma_20', 'sma_50', 'dow']
        feature_cols: List[str] = [col for col in list(features_df.columns) if col not in exclude_cols]

        # Align features with training features (like V1 does)
        if self.feature_names:
            # Add missing features with 0
            for feat in self.feature_names:
                if feat not in latest_df.columns:
                    latest_df[feat] = 0
            # Reorder to match training
            X = latest_df[self.feature_names]
        else:
            X = latest_df[feature_cols]

        # Clean data
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        try:
            # Predict using XGBoost booster (like V1 does)
            # Create DMatrix
            dmat = xgb.DMatrix(X, feature_names=self.feature_names if self.feature_names else feature_cols)

            # Get probability from booster
            if isinstance(self.model, xgb.XGBClassifier):
                # XGBClassifier - use get_booster()
                booster = self.model.get_booster()
                confidence = booster.predict(dmat)[0]
            elif isinstance(self.model, xgb.Booster):
                # Direct Booster object
                confidence = self.model.predict(dmat)[0]
            else:
                return None

            # Get ATR for dynamic SL/TP (from last row of original dataframe)
            latest_row = features_df.iloc[-1]
            has_atr = 'atr_pct' in list(features_df.columns)
            if has_atr:
                atr_value = float(latest_row['atr_pct'])
                atr_pct = atr_value if not pd.isna(atr_value) else 0.03
            else:
                atr_pct = 0.03

            # V1 default: 2.0x SL, 5.0x TP, 2.5x Trailing (from metadata or default)
            sl_mult = self.metadata.get('sl_atr_mult', 2.0)
            tp_mult = self.metadata.get('tp_atr_mult', 5.0)
            trail_mult = self.metadata.get('trailing_stop_atr_mult', 2.5)

            # Calculate actual SL/TP/Trailing percentages
            sl_pct = atr_pct * sl_mult
            tp_pct = atr_pct * tp_mult
            trail_pct = atr_pct * trail_mult

            # Determine order type based on confidence
            # Market order: very high confidence (>= 0.85)
            # Limit order: moderate confidence (0.55-0.85)
            if confidence >= 0.85:
                order_type = "MARKET"
                entry_pct = 0.0  # Enter at market price
            else:
                order_type = "LIMIT"
                # Limit order: enter 1x ATR below current (conservative entry)
                entry_pct = atr_pct * 1.0

            return {
                'confidence': confidence,
                'sl_pct': sl_pct,
                'tp_pct': tp_pct,
                'trail_pct': trail_pct,
                'sl_mult': sl_mult,
                'tp_mult': tp_mult,
                'trail_mult': trail_mult,
                'atr_pct': atr_pct,
                'order_type': order_type,
                'entry_pct': entry_pct
            }

        except Exception as e:
            print(f"XGBoost prediction failed: {e}")
            return None

    def get_threshold(self):
        """Get signal threshold from metadata or use default."""
        return self.metadata.get('signal_threshold', 0.55)

    def get_metadata(self):
        """Return model metadata."""
        return self.metadata
