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
    min_data_points: int = 100  # Minimum data points for calculation


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
    # Training
    training_window: int = 252  # Rolling window size (1 year)
    validation_split: float = 0.2
    
    # Prediction target (aligned with max_holding_days for day trading)
    target_horizon: int = 5  # Predict 5-day forward return (matches max hold)
    
    # XGBoost parameters (Gen 5 Ultimate Specs)
    n_estimators: int = 150
    max_depth: int = 6
    learning_rate: float = 0.1

    min_child_weight: int = 3
    use_gpu: bool = False
    
    # Feature settings
    feature_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])


@dataclass
class ExitConfig:
    """Exit strategy configuration"""
    # ATR-based stops
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    trailing_stop_atr_mult: float = 1.5
    
    # Time-based exit
    max_holding_days: int = 5
    
    # Fixed stops (fallback)
    max_loss_pct: float = 0.08  # 8% max loss
    min_profit_pct: float = 0.02  # Minimum 2% to consider partial exit


@dataclass
class PortfolioConfig:
    """Portfolio management configuration"""
    # Total portfolio value for sizing calculations
    total_value: float = 100_000_000  # Default 100M IDR
    
    # Position sizing
    max_positions: int = 10
    base_position_pct: float = 0.10  # 10% per position
    max_sector_exposure: float = 0.30  # 30% max per sector
    
    # Liquidity filter
    min_avg_volume: int = 1_000_000  # 1M shares minimum
    min_market_cap: float = 1e12  # 1 Trillion IDR minimum
    
    # Risk management
    max_portfolio_volatility: float = 0.25  # 25% annual vol target
    max_correlation: float = 0.7  # Avoid highly correlated positions


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
