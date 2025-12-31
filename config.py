"""
IHSG Quantitative Trading Model Configuration
"""
from dataclasses import dataclass, field
from typing import List
import os
from data.universe import IDX_UNIVERSE


@dataclass
class DataConfig:
    """Data fetching configuration"""
    # Stock universe - Comprehensive Liquid IHSG Universe (IDX80 + Kompas100 + Liquid Growth)
    stock_universe: List[str] = field(default_factory=lambda: IDX_UNIVERSE)
    
    # Data storage
    db_path: str = "data/ihsg_trading.db"
    
    # Historical data settings
    lookback_days: int = 365 * 5  # 5 years of history
    min_data_points: int = 252  # Minimum data points for calculation


@dataclass
class SignalConfig:
    """Signal generation parameters"""
    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # Mean reversion
    zscore_period: int = 20
    zscore_entry_threshold: float = 2.0
    
    # Momentum
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class MLConfig:
    """Machine learning model configuration"""
    # Training windows (5 years total - using more of your 5-year historical data)
    training_window: int = 1008  # 4 years of trading days for training
    validation_window: int = 252  # 1 year of trading days for validation
    validation_split: float = 0.2  # Deprecated, use validation_window instead

    # Prediction target (aligned with max_holding_days for day trading)
    target_horizon: int = 5  # Predict 5-day forward return (matches max hold)

    # XGBoost parameters (Conservative Defaults with Strong Regularization)
    # Note: Based on training results, 50 trees is the sweet spot before overfitting
    # Regularization allows using more trees without overfitting
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.1
    min_child_weight: int = 5

    # Regularization parameters (prevent overfitting)
    subsample: float = 0.8  # Use 80% of data per tree (randomness prevents overfitting)
    colsample_bytree: float = 0.7  # Use 70% of features per tree
    gamma: float = 0.2  # Minimum loss reduction required to split (conservative)
    reg_alpha: float = 1.0       # Stronger L1 regularization (feature selection)
    reg_lambda: float = 2.0      # Stronger L2 regularization

    use_gpu: bool = False

    # Feature settings
    feature_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 15, 20])

@dataclass
class ExitConfig:
    """Exit strategy configuration"""
    stop_loss_atr_mult: float = 2.0       # Tighter stop (2.0x is standard for swing)
    take_profit_atr_mult: float = 5.0     # Reduced from 10.0. (4-6x is a "Home Run" in 5 days)
    trailing_stop_atr_mult: float = 2.5   # Trail loosely to let winners run

    # Time-based exit (hold longer for bigger moves)
    max_holding_days: int = 5

    # Fixed stops (fallback for extreme cases)
    max_loss_pct: float = 0.08    # Tighten max loss to 8% (preservation is key)
    min_profit_pct: float = 0.15  # Realistic 15% upside target for 1 week
 
    signal_threshold: float = 0.40  # Minimum ML signal to enter/hold position

@dataclass
class PortfolioConfig:
    """Portfolio management configuration"""
    # Total portfolio value for sizing calculations
    total_value: float = 100_000_000  # Default 100M IDR
    
    # Position sizing
    max_positions: int = 8             # Reduced: Focus capital on best ideas
    base_position_pct: float = 0.125   # 12.5% per trade
    max_sector_exposure: float = 0.25  # 25% max per sector
    
    # Liquidity filter
    min_avg_volume: int = 1_000_000  # 2M shares minimum
    min_market_cap: float = 2e12  # 2 Trillion IDR minimum
    
    # Risk management
    max_portfolio_volatility: float = 0.20  # 20% annual vol target
    max_correlation: float = 0.6  # Avoid highly correlated positions


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    
    # Paths
    reports_dir: str = "reports"
    models_dir: str = "models"
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.data.db_path) or "data", exist_ok=True)


# Global config instance
config = Config()
