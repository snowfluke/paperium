"""
Statistical Signals Module
Mean reversion, momentum, and GARCH volatility models
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StatisticalSignals:
    """
    Statistical arbitrage signals inspired by Renaissance Technologies approach.
    Implements mean reversion, momentum, and volatility regime detection.
    """
    
    def __init__(self, config=None):
        """
        Initialize with configuration.
        
        Args:
            config: SignalConfig object (optional)
        """
        if config:
            self.zscore_period = config.zscore_period
            self.zscore_threshold = config.zscore_entry_threshold
            self.momentum_periods = config.momentum_periods
        else:
            self.zscore_period = 20
            self.zscore_threshold = 2.0
            self.momentum_periods = [5, 10, 20]
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all statistical signals.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added statistical signal columns
        """
        df = df.copy()
        
        df = self.add_zscore(df)
        df = self.add_momentum(df)
        df = self.add_volatility(df)
        df = self.add_returns(df)
        df = self.add_garch_volatility(df)
        
        return df
    
    def add_zscore(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate Z-score for mean reversion signals.
        
        Z-score = (price - mean) / std
        High positive Z-score = overbought (sell signal)
        High negative Z-score = oversold (buy signal)
        """
        period = period or self.zscore_period
        
        rolling_mean = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        
        df['zscore'] = (df['close'] - rolling_mean) / rolling_std.replace(0, np.inf)
        
        # Mean reversion signals
        df['mean_rev_buy'] = (df['zscore'] < -self.zscore_threshold).astype(int)
        df['mean_rev_sell'] = (df['zscore'] > self.zscore_threshold).astype(int)
        
        # Z-score normalized to -1 to 1 range for scoring
        df['zscore_norm'] = -np.clip(df['zscore'] / 3, -1, 1)  # Negative because low z = buy signal
        
        return df
    
    def add_momentum(self, df: pd.DataFrame, periods: Optional[list] = None) -> pd.DataFrame:
        """
        Calculate momentum indicators (rate of change).
        
        Momentum = (price_t - price_t-n) / price_t-n
        """
        periods = periods or self.momentum_periods
        
        for period in periods:
            df[f'momentum_{period}'] = df['close'].pct_change(periods=period)
            df[f'momentum_{period}_rank'] = df[f'momentum_{period}'].rolling(
                window=60, min_periods=20
            ).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10))
        
        # Composite momentum score
        mom_cols = [f'momentum_{p}' for p in periods]
        df['momentum_composite'] = df[mom_cols].mean(axis=1)
        
        # Normalize to -1 to 1
        df['momentum_score'] = np.clip(df['momentum_composite'] * 10, -1, 1)
        
        return df
    
    def add_volatility(self, df: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate historical volatility measures.
        
        Volatility = std(returns) * sqrt(252)  # Annualized
        """
        # Daily returns
        returns = df['close'].pct_change()
        
        for period in periods:
            # Historical volatility (annualized)
            df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
        
        # Volatility regime (compare short-term to long-term vol)
        df['vol_regime'] = df['volatility_5'] / df['volatility_20'].replace(0, np.inf)
        df['high_vol_regime'] = (df['vol_regime'] > 1.2).astype(int)
        df['low_vol_regime'] = (df['vol_regime'] < 0.8).astype(int)
        
        return df
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various return calculations."""
        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(5)
        df['return_20d'] = df['close'].pct_change(20)
        
        # Log returns (for statistical analysis)
        ratio = (df['close'] / df['close'].shift(1)).replace([0, np.inf, -np.inf], np.nan).fillna(1.0)
        df['log_return'] = np.log(ratio.abs())


        
        # Cumulative returns
        df['cum_return_20d'] = df['return_1d'].rolling(window=20).sum()
        
        return df
    
    def add_garch_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate GARCH(1,1) volatility forecast.
        
        Simplified GARCH without using arch library for speed.
        Uses exponentially weighted variance as approximation.
        """
        returns = df['close'].pct_change().fillna(0)
        
        # EWMA variance (GARCH approximation)
        lambda_param = 0.94  # RiskMetrics decay factor
        ewma_var = returns.ewm(alpha=1-lambda_param, adjust=False).var()
        
        df['garch_vol'] = np.sqrt(ewma_var * 252)  # Annualized
        
        # Volatility forecast vs current (for regime detection)
        df['vol_expanding'] = (df['garch_vol'] > df['garch_vol'].shift(5)).astype(int)
        df['vol_contracting'] = (df['garch_vol'] < df['garch_vol'].shift(5)).astype(int)
        
        # Full GARCH(1,1) using arch library (if available, slower but more accurate)
        try:
            df = self._fit_garch_model(df, returns)
        except (ImportError, Exception) as e:
            logger.debug(f"GARCH model not fitted: {e}")
        
        return df
    
    def _fit_garch_model(self, df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """
        Fit full GARCH(1,1) model using arch library.
        
        This is the more accurate version used by quant funds.
        """
        from arch import arch_model
        
        # Only fit if we have enough data
        if len(returns.dropna()) < 100:
            return df
        
        # Scale returns (GARCH expects percentage returns)
        scaled_returns = returns.dropna() * 100
        
        # Fit GARCH(1,1) with strict warning suppression to keep console clean
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)
        
        # Get conditional volatility
        cond_vol = result.conditional_volatility / 100  # Scale back

        # Forecast next-day volatility
        forecast = result.forecast(horizon=1)

        # Align with original DataFrame - convert to Series if it's an array
        if isinstance(cond_vol, pd.Series):
            df.loc[cond_vol.index, 'garch_cond_vol'] = cond_vol.values
        else:
            # If it's an ndarray, convert to Series first
            cond_vol_series = pd.Series(cond_vol, index=scaled_returns.index)
            df.loc[cond_vol_series.index, 'garch_cond_vol'] = cond_vol_series.values
        
        return df
    
    def get_mean_reversion_score(self, row: pd.Series) -> float:
        """
        Calculate mean reversion score.
        
        Returns:
            Score between -1 (strong sell) and 1 (strong buy)
        """
        if 'zscore_norm' in row and pd.notna(row['zscore_norm']):
            return row['zscore_norm']
        return 0.0
    
    def get_momentum_score(self, row: pd.Series) -> float:
        """
        Calculate momentum score.
        
        Returns:
            Score between -1 and 1
        """
        if 'momentum_score' in row and pd.notna(row['momentum_score']):
            return row['momentum_score']
        return 0.0
    
    def get_volatility_score(self, row: pd.Series) -> float:
        """
        Calculate volatility-based score.
        
        Low volatility with positive momentum = bullish
        High volatility regime = reduce position sizes
        
        Returns:
            Score between -1 and 1
        """
        score = 0.0
        
        if 'vol_regime' in row and pd.notna(row['vol_regime']):
            # Lower volatility relative to history is favorable
            if row['vol_regime'] < 0.8:
                score += 0.3
            elif row['vol_regime'] > 1.5:
                score -= 0.3
        
        if 'vol_contracting' in row and row['vol_contracting']:
            score += 0.2  # Vol contraction is bullish
        elif 'vol_expanding' in row and row['vol_expanding']:
            score -= 0.2  # Vol expansion is cautious
        
        return np.clip(score, -1, 1)


class PairTrading:
    """
    Statistical arbitrage through pair trading.
    Identifies correlated stocks and trades mean reversion of the spread.
    """
    
    def __init__(self, lookback: int = 60, zscore_threshold: float = 2.0):
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold
    
    def find_correlated_pairs(
        self, 
        price_data: pd.DataFrame, 
        min_correlation: float = 0.8
    ) -> list:
        """
        Find highly correlated stock pairs.
        
        Args:
            price_data: DataFrame with 'ticker' and 'close' columns
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of tuples (ticker1, ticker2, correlation)
        """
        # Pivot to get price matrix
        pivot = price_data.pivot(columns='ticker', values='close')
        
        # Calculate returns
        returns = pivot.pct_change().dropna()
        
        # Correlation matrix
        corr_matrix = returns.corr()
        
        # Find pairs above threshold
        pairs = []
        tickers = corr_matrix.columns.tolist()
        
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i+1:]:
                corr_val = corr_matrix.loc[t1, t2]
                # Convert to float, handling numpy/pandas scalar types
                if isinstance(corr_val, (int, float, np.integer, np.floating)):
                    corr = float(corr_val)
                elif isinstance(corr_val, (complex, np.complexfloating)):
                    corr = float(corr_val.real)  # Handle complex numbers by taking real part
                else:
                    # Skip non-numeric values
                    continue
                if corr >= min_correlation:
                    pairs.append((t1, t2, corr))
        
        # Sort by correlation
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs
    
    def calculate_spread_zscore(
        self, 
        prices1: pd.Series, 
        prices2: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate spread and its Z-score between two price series.
        
        Returns:
            Tuple of (spread, zscore)
        """
        # Log price ratio (spread) - Safe log
        ratio = (prices1 / prices2).replace([0, np.inf, -np.inf], np.nan).fillna(1.0)
        spread = pd.Series(np.log(ratio.abs()), index=ratio.index)


        # Rolling Z-score
        mean = spread.rolling(window=self.lookback).mean()
        std = spread.rolling(window=self.lookback).std()
        zscore = (spread - mean) / std.replace(0, np.inf)

        return spread, zscore
