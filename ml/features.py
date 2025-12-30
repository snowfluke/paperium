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

    # GEN 7 Feature Set: Adds Intraday Behavior Proxies
    GEN7_FEATURES = LEGACY_46_FEATURES + [
        'upper_shadow_ratio',    # Upper wick / body - detects rejection at highs
        'gap_size',              # Gap magnitude (positive or negative)
        'gap_followthrough',     # Close vs Open after gap - measures gap strength
        'intraday_momentum',     # Close position in daily range - strength indicator
        'fade_signal',           # Combined: gap exhaustion + weak close + upper shadow
    ]

    # GEN 8 Feature Set: Adds TRUE Hour-0 metrics from intraday data (optional)
    GEN8_FEATURES = GEN7_FEATURES + [
        'h0_spike_pct',          # Actual 9-10 AM spike from hourly data
        'h0_fade_pct',           # Actual 10-11 AM fade from hourly data
        'h0_net_pct',            # Net Hour-0 movement
        'h0_spike_is_day_high',  # Whether 10 AM was day's high
        'h0_spike_to_close',     # Spike to close relationship
    ]

    # GEN 9 Feature Set: Ultimate - Adds S/D zones + Microstructure + Order Flow
    GEN9_FEATURES = GEN8_FEATURES + [
        # Supply/Demand Zone Features (9 features)
        'sd_score',              # Overall S/D voting score (-1 to +1)
        'sd_demand_distance',    # Distance to nearest demand zone
        'sd_demand_strength',    # Strength of nearest demand zone
        'sd_supply_distance',    # Distance to nearest supply zone
        'sd_supply_strength',    # Strength of nearest supply zone
        'sd_demand_fresh',       # Is demand zone fresh (untested)?
        'sd_supply_fresh',       # Is supply zone fresh (untested)?
        'sd_zone_count',         # Total number of active zones
        'sd_net_strength',       # Net strength (demand - supply)

        # Market Microstructure (10 features)
        'volume_profile_poc',    # Point of Control (price with most volume)
        'volume_imbalance',      # Buy vs Sell volume proxy
        'price_efficiency',      # Price movement per unit volume
        'volume_weighted_price', # VWAP deviation
        'large_trade_ratio',     # Ratio of large trades (>2x avg)
        'volume_momentum',       # Rate of change in volume
        'bid_ask_spread_proxy',  # High-Low / Close (liquidity proxy)
        'tick_direction',        # Net up ticks vs down ticks
        'absorption_score',      # Price resistance to volume
        'exhaustion_signal',     # High volume with small price move

        # Order Flow Proxies (6 features)
        'buying_pressure',       # Up volume / Total volume
        'selling_pressure',      # Down volume / Total volume
        'delta_volume',          # Net volume (buy - sell)
        'cumulative_delta',      # Cumulative delta over 5 days
        'volume_at_price_high',  # Volume when price = session high
        'volume_at_price_low',   # Volume when price = session low
    ]

    def __init__(self, config=None, use_gen7_features=True, use_hour0_features='auto', use_gen9_features=False):
        """
        Initialize feature engineer.

        Args:
            config: MLConfig object (optional)
            use_gen7_features: If True, use GEN7 feature set with Session-1 features (default True)
            use_hour0_features: 'auto' (detect from DB), True, or False (default 'auto')
            use_gen9_features: If True, use GEN9 with S/D zones + microstructure (default False)
        """
        if config:
            self.feature_lags = config.feature_lags
            self.target_horizon = getattr(config, 'target_horizon', 5)  # Default 5 for day trading
        else:
            self.feature_lags = [1, 2, 3, 5, 10, 20]
            self.target_horizon = 5  # 5-day forward prediction for day trading

        # Initialize sub-modules for internal indicator calculation
        self.technical = TechnicalIndicators()

        # Auto-detect Hour-0 features if set to 'auto'
        if use_hour0_features == 'auto':
            use_hour0_features = self._check_hour0_available()

        # Initialize S/D detector for GEN9
        self.use_gen9 = use_gen9_features
        if use_gen9_features:
            from signals.supply_demand import SupplyDemandDetector
            self.sd_detector = SupplyDemandDetector()

        # Select feature set
        if use_gen9_features:
            self.feature_set = self.GEN9_FEATURES
            self.use_hour0 = use_hour0_features
        elif use_hour0_features:
            self.feature_set = self.GEN8_FEATURES
            self.use_hour0 = True
        elif use_gen7_features:
            self.feature_set = self.GEN7_FEATURES
            self.use_hour0 = False
        else:
            self.feature_set = self.LEGACY_46_FEATURES
            self.use_hour0 = False

    # Class-level cache for Hour-0 availability check
    _hour0_cache = None

    def _check_hour0_available(self) -> bool:
        """Check if Hour-0 metrics table exists in database (cached)."""
        # Use cached result if available
        if FeatureEngineer._hour0_cache is not None:
            return FeatureEngineer._hour0_cache

        try:
            from data.storage import DataStorage
            from config import config
            import sqlite3

            storage = DataStorage(config.data.db_path)
            with sqlite3.connect(storage.db_path) as conn:
                table_check = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='hour0_metrics'"
                ).fetchone()
                result = table_check is not None
                # Cache the result
                FeatureEngineer._hour0_cache = result
                return result
        except Exception:
            FeatureEngineer._hour0_cache = False
            return False
    
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

        # 5. Hour-0 features (if enabled and data available)
        if self.use_hour0:
            df = self._add_hour0_features(df)

        # 6. GEN9 Advanced Features (S/D zones + Microstructure + Order Flow)
        if self.use_gen9:
            df = self._add_supply_demand_features(df)
            df = self._add_microstructure_features(df)
            df = self._add_order_flow_features(df)

        # 7. Drop rows with NaN
        df = df.dropna()
        
        # 6. Ensure EXACT feature list and order
        X = df[self.feature_set]
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

        # --- INTRADAY BEHAVIOR PROXIES (GEN 7) ---
        # Hypothesis: Stocks with weak intraday behavior (pop then fade) tend to underperform
        # Strategy: Use daily OHLC patterns to detect weak intraday structure

        # Note: We trade EOD → hold next day, so we predict "tomorrow's outcome" not "today's intraday"

        # 1. Upper Shadow Ratio: How much of the candle is upper wick
        # High value = price rejected highs (likely faded from early spike)
        # This indicates weak buying pressure / profit-taking
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        body_size = (df[['open', 'close']].max(axis=1) - df[['open', 'close']].min(axis=1)).replace(0, 1)
        df['upper_shadow_ratio'] = upper_shadow / body_size

        # 2. Gap Exhaustion: Large gap that fails to follow through
        # Gap up + close near/below open = exhaustion (bad for tomorrow)
        df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, 1)
        df['gap_followthrough'] = (df['close'] - df['open']) / df['open'].replace(0, 1)

        # 3. Intraday Strength: Where close is within the day's range
        # Already have 'close_position' from legacy features, but add momentum version
        # High value = closed near high (strength), Low value = closed near low (weakness)
        df['intraday_momentum'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)

        # 4. Fade Signal: Combined metric for gap exhaustion
        # High when: Large gap + weak close + upper shadow
        # This suggests: "Gapped up, spiked early, faded, closed weak" → AVOID tomorrow
        df['fade_signal'] = (df['gap_size'].abs() * df['upper_shadow_ratio'] * (1 - df['intraday_momentum'])).fillna(0)

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

    def _add_hour0_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Hour-0 features from intraday metrics database.
        Merges actual 9-11 AM metrics if available.
        """
        import time
        start_time = time.time()

        if 'ticker' not in df.columns or 'date' not in df.columns:
            # Fill with zeros if can't merge
            logger.debug(f"Hour-0: No ticker/date columns, filling with zeros")
            for feat in ['h0_spike_pct', 'h0_fade_pct', 'h0_net_pct', 'h0_spike_is_day_high', 'h0_spike_to_close']:
                df[feat] = 0.0
            return df

        try:
            from data.storage import DataStorage
            from config import config
            import sqlite3

            storage = DataStorage(config.data.db_path)
            logger.debug(f"Hour-0: Starting feature merge for {len(df)} rows")

            # Check if hour0_metrics table exists
            with sqlite3.connect(storage.db_path) as conn:
                table_check = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='hour0_metrics'"
                ).fetchone()

                if not table_check:
                    # Table doesn't exist yet - fill with zeros
                    for feat in ['h0_spike_pct', 'h0_fade_pct', 'h0_net_pct', 'h0_spike_is_day_high', 'h0_spike_to_close']:
                        df[feat] = 0.0
                    return df

            # Get ticker for this dataframe
            ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else None
            if not ticker:
                logger.debug(f"Hour-0: No ticker found, filling with zeros")
                for feat in ['h0_spike_pct', 'h0_fade_pct', 'h0_net_pct', 'h0_spike_is_day_high', 'h0_spike_to_close']:
                    df[feat] = 0.0
                return df

            # Load Hour-0 metrics for this ticker
            query_start = time.time()
            with sqlite3.connect(storage.db_path) as conn:
                h0_data = pd.read_sql_query(
                    "SELECT * FROM hour0_metrics WHERE ticker = ?",
                    conn,
                    params=(ticker,)
                )
            logger.debug(f"Hour-0: DB query for {ticker} took {time.time() - query_start:.3f}s, got {len(h0_data)} rows")

            if h0_data.empty:
                # No data for this ticker - fill with zeros
                for feat in ['h0_spike_pct', 'h0_fade_pct', 'h0_net_pct', 'h0_spike_is_day_high', 'h0_spike_to_close']:
                    df[feat] = 0.0
                return df

            # Merge with main dataframe
            merge_start = time.time()
            h0_data['date'] = pd.to_datetime(h0_data['date'])
            df['date'] = pd.to_datetime(df['date'])

            df = df.merge(
                h0_data[['date', 'h0_spike_pct', 'h0_fade_pct', 'h0_net_pct', 'h0_spike_is_day_high', 'h0_spike_to_close']],
                on='date',
                how='left'
            )
            logger.debug(f"Hour-0: Merge took {time.time() - merge_start:.3f}s")

            # Fill missing values with 0
            for feat in ['h0_spike_pct', 'h0_fade_pct', 'h0_net_pct', 'h0_spike_is_day_high', 'h0_spike_to_close']:
                df[feat] = df[feat].fillna(0.0)

            logger.debug(f"Hour-0: Total feature merge time: {time.time() - start_time:.3f}s")

        except Exception as e:
            logger.warning(f"Failed to add Hour-0 features: {e}")
            # Fill with zeros on error
            for feat in ['h0_spike_pct', 'h0_fade_pct', 'h0_net_pct', 'h0_spike_is_day_high', 'h0_spike_to_close']:
                df[feat] = 0.0

        return df

    def _add_supply_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add supply/demand zone features using the SupplyDemandDetector.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with S/D features added
        """
        if not hasattr(self, 'sd_detector'):
            # Fallback: fill with zeros if detector not initialized
            for feat in ['sd_score', 'sd_demand_distance', 'sd_demand_strength',
                        'sd_supply_distance', 'sd_supply_strength', 'sd_demand_fresh',
                        'sd_supply_fresh', 'sd_zone_count', 'sd_net_strength']:
                df[feat] = 0.0
            return df

        try:
            # Get S/D features for current data
            sd_features = self.sd_detector.get_zone_features(df)

            # Add features to dataframe
            for key, value in sd_features.items():
                df[key] = value

        except Exception as e:
            logger.warning(f"Failed to add S/D features: {e}")
            # Fill with zeros on error
            for feat in ['sd_score', 'sd_demand_distance', 'sd_demand_strength',
                        'sd_supply_distance', 'sd_supply_strength', 'sd_demand_fresh',
                        'sd_supply_fresh', 'sd_zone_count', 'sd_net_strength']:
                df[feat] = 0.0

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features.

        These features capture market liquidity, efficiency, and institutional behavior.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with microstructure features
        """
        # 1. Volume Profile - Point of Control (price level with most volume)
        # Proxy: Use weighted average of prices by volume over rolling window
        window = 20
        df['volume_profile_poc'] = (df['close'] * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
        df['volume_profile_poc'] = (df['close'] - df['volume_profile_poc']) / df['close']  # Deviation from POC

        # 2. Volume Imbalance (Buy vs Sell pressure proxy)
        # When close > open: buying pressure, when close < open: selling pressure
        # Weight by volume
        imbalance = ((df['close'] - df['open']) / df['open'].replace(0, 1)) * df['volume']
        df['volume_imbalance'] = imbalance.rolling(window).sum() / df['volume'].rolling(window).sum()

        # 3. Price Efficiency (Price movement per unit of volume)
        # Low efficiency = absorption (lots of volume, little price movement)
        price_change = df['close'].diff().abs()
        df['price_efficiency'] = (price_change / df['volume'].replace(0, 1)).rolling(window).mean()

        # 4. Volume Weighted Average Price (VWAP) Deviation
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
        df['volume_weighted_price'] = (df['close'] - vwap) / df['close']

        # 5. Large Trade Ratio (Proxy: days with volume > 2x average)
        avg_volume = df['volume'].rolling(window).mean()
        df['large_trade_ratio'] = (df['volume'] / avg_volume.replace(0, 1)).rolling(10).apply(lambda x: (x > 2.0).mean())

        # 6. Volume Momentum (Rate of change in volume)
        df['volume_momentum'] = df['volume'].pct_change(5)

        # 7. Bid-Ask Spread Proxy (High-Low range as % of close)
        df['bid_ask_spread_proxy'] = (df['high'] - df['low']) / df['close']

        # 8. Tick Direction (Net up closes vs down closes)
        df['tick_direction'] = (df['close'] > df['close'].shift(1)).astype(int).rolling(window).mean()

        # 9. Absorption Score (Price resistance to volume)
        # High volume + small price change = absorption (institutional accumulation/distribution)
        volume_zscore = (df['volume'] - df['volume'].rolling(window).mean()) / df['volume'].rolling(window).std()
        price_change_norm = df['close'].pct_change().abs()
        df['absorption_score'] = volume_zscore * (1 - price_change_norm.rolling(5).mean())

        # 10. Exhaustion Signal (High volume with small range)
        # High volume but price doesn't move = exhaustion
        volume_percentile = df['volume'].rolling(window).apply(lambda x: (x.iloc[-1] > x).mean())
        range_percentile = df['hl_range'].rolling(window).apply(lambda x: (x.iloc[-1] > x).mean() if 'hl_range' in df.columns else 0.5)
        df['exhaustion_signal'] = volume_percentile - range_percentile  # Positive = exhaustion

        return df

    def _add_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add order flow proxy features.

        These approximate buying/selling pressure using OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with order flow features
        """
        # 1. Buying Pressure (Proxy: Volume when close > open)
        buy_volume = df['volume'] * (df['close'] > df['open']).astype(int)
        df['buying_pressure'] = buy_volume.rolling(10).sum() / df['volume'].rolling(10).sum()

        # 2. Selling Pressure (Proxy: Volume when close < open)
        sell_volume = df['volume'] * (df['close'] < df['open']).astype(int)
        df['selling_pressure'] = sell_volume.rolling(10).sum() / df['volume'].rolling(10).sum()

        # 3. Delta Volume (Net buying - selling volume)
        # Weighted by where close is in the range
        close_position = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
        df['delta_volume'] = df['volume'] * (2 * close_position - 1)  # -1 to +1 weighting

        # 4. Cumulative Delta (Cumulative sum of delta volume)
        df['cumulative_delta'] = df['delta_volume'].rolling(5).sum()

        # 5. Volume at Price High (Volume when price = session high)
        # Proxy: Volume when close is within 1% of high
        at_high = ((df['high'] - df['close']) / df['close'] < 0.01).astype(int)
        df['volume_at_price_high'] = (df['volume'] * at_high).rolling(10).sum() / df['volume'].rolling(10).sum()

        # 6. Volume at Price Low (Volume when price = session low)
        # Proxy: Volume when close is within 1% of low
        at_low = ((df['close'] - df['low']) / df['close'] < 0.01).astype(int)
        df['volume_at_price_low'] = (df['volume'] * at_low).rolling(10).sum() / df['volume'].rolling(10).sum()

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

        if self.use_hour0:
            df = self._add_hour0_features(df)

        if self.use_gen9:
            df = self._add_supply_demand_features(df)
            df = self._add_microstructure_features(df)
            df = self._add_order_flow_features(df)

        X = df[self.feature_set].fillna(0)

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
