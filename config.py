"""
IHSG Quantitative Trading Model Configuration
Refactored for LSTM + Triple Barrier Labeling (Paper Implementation)
"""
from dataclasses import dataclass, field
from typing import List
import os
from data.universe import IDX_UNIVERSE


@dataclass
class DataConfig:
    """Data fetching configuration"""
    # Stock universe
    stock_universe: List[str] = field(default_factory=lambda: IDX_UNIVERSE)
    
    # Data storage
    db_path: str = "data/ihsg_trading.db"
    
    # Historical data settings
    lookback_days: int = 365 * 5  # 5 years
    min_data_points: int = 252
    
    # Sequence generation
    window_size: int = 100  # From Paper: Optimal Window Size


@dataclass
class MLConfig:
    """Machine learning model configuration (LSTM)"""
    # Model Architecture (Paper: Hidden Size 8 was optimal)
    input_size: int = 5  # Open, High, Low, Close, Volume
    hidden_size: int = 8
    num_layers: int = 2  # Starting with 2 stacked LSTMs
    dropout: float = 0.0
    
    # Training
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 50
    patience: int = 10  # Early stopping
    
    # Triple Barrier Labeling (Optimized for IHSG)
    # Found via optimization: Horizon=5, Barrier=3.0%
    tbl_horizon: int = 5
    tbl_barrier: float = 0.03
    num_classes: int = 3  # 0: Loss, 1: Hold, 2: Profit


@dataclass
class ExitConfig:
    """Exit strategy configuration"""
    # Legacy ATR-based exits (kept for PositionManager compatibility)
    # TBL barrier is 3%, so we map roughly to that. 
    # If ATR is ~2%, then 1.5x ATR = 3%.
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 1.5
    signal_threshold: float = 0.5


@dataclass
class PortfolioConfig:
    """Portfolio simulation settings"""
    total_value: float = 100_000_000.0  # Default 100M IDR
    max_positions: int = 10
    risk_per_trade: float = 0.02  # 2% Risk per trade (Optional usage)


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
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
