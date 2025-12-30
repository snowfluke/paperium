# Paperium Trading System - Project Knowledge

**Last Updated:** 2025-12-30
**Status:** Production-ready with signal-strength-based position sizing

## System Overview

Paperium is a quantitative trading system for the Indonesia Stock Exchange (IHSG) that uses machine learning (XGBoost) to predict 5-day forward returns and generate daily trading signals. The system has been refactored to eliminate model regression caused by complex optimization schemes.

## Core Philosophy

**"Keep it simple - complexity introduces failure modes"**

The system was redesigned around these principles:

1. Fixed configuration over optimization
2. Single universal feature set (no versioning)
3. Fresh training without warm starts
4. Balanced regularization without aggressive penalties
5. Focus on win rate - profitability follows naturally

## Architecture

### Key Components

1. **Data Layer** (`data/`)

   - Storage: SQLite database (`data/ihsg_trading.db`)
   - Universe: ~956 tickers (IDX80, Kompas100, growth sectors)
   - History: 5 years of OHLCV data
   - Index: IHSG for crash detection

2. **Feature Engineering** (`ml/features.py`)

   - Universal 51-feature set (NO generation versioning)
   - Base 46 features: technical indicators, returns, volatility
   - Plus 5 intraday behavior proxies: upper_shadow_ratio, gap_size, gap_followthrough, intraday_momentum, fade_signal
   - Target: 5-day forward return direction (binary classification)

3. **Model** (`ml/model.py`)

   - Single global XGBoost classifier
   - Trained on pooled data from all tickers
   - Noise downsampling: 50% of boring days (|return| < 0.5%) removed
   - Magnitude weighting: w = 1.0 + |return| \* 10
   - Fresh training each run (no incremental updates)

4. **Backtesting** (`scripts/eval.py`)

   - Walk-forward simulation
   - ATR-based dynamic SL/TP
   - Position sizing: 10% per position, max 10 concurrent
   - Time-based exit: max 5 days hold
   - IHSG crash filter via RegimeDetector

5. **Production Scripts**
   - `train.py`: Single training run (ONLY training script)
   - `eval.py`: Backtest with historical data
   - `morning_signals.py`: Generate pre-market signals
   - `eod_retrain.py`: End-of-day position evaluation (retraining disabled by default)

## Critical Configuration

All configuration is centralized in `config.py`:

### Training Windows (5 years total)

```python
training_window: int = 1008  # 4 years of trading days (maximum historical data)
validation_window: int = 252  # 1 year of trading days
```

### Exit Strategy (FIXED - not optimized)

```python
stop_loss_atr_mult: float = 2.0    # 2.0x ATR stop loss (tighter, standard for swing)
take_profit_atr_mult: float = 5.0  # 5.0x ATR take profit (realistic "home run" target)
trailing_stop_atr_mult: float = 2.5 # Trail loosely to let winners run
signal_threshold: float = 0.55     # Selective: 85%+ win rate
max_holding_days: int = 5          # Short-term swing trades
max_loss_pct: float = 0.08         # 8% max loss (tighter risk management)
min_profit_pct: float = 0.15       # 15% realistic upside target
```

**Performance Targets (Balanced Conservative Strategy):**

- Win Rate: 85%+ (selective 0.55 threshold)
- Average PnL per Trade: 15% (realistic 1-week target)
- W/L Ratio: 2.5-3.0x (balanced risk/reward with 2:5 SL/TP ratio)
- Trade Frequency: 10-15 trades/month (selective quality)
- Max Holding: 5 days (short-term swing trading)

### XGBoost Parameters (With Very Strong Regularization - Conservative)

```python
n_estimators: int = 500        # Many weak learners (low learning rate compensates)
max_depth: int = 4             # Shallower trees (was 5) - prevents overfitting
learning_rate: float = 0.05    # Very slow learning (was 0.1) - gradual improvement
min_child_weight: int = 5      # Higher threshold (was 3) - more conservative splits
subsample: float = 0.7         # 70% data per tree (was 0.8) - more randomness
colsample_bytree: float = 0.7  # 70% features per tree (was 0.8) - more diversity
gamma: float = 0.2             # Higher split threshold (was 0.1) - very conservative
reg_alpha: float = 1.0         # Stronger L1 (was 0.1) - aggressive feature selection
reg_lambda: float = 2.0        # Stronger L2 (was 1.0) - prevent large weights
feature_lags: [1,2,3,5,10,15,20] # Added 15-day lag
```

**Strategy:** Many shallow, slow-learning trees with strong regularization prevents overfitting while maintaining predictive power.

**Note:** Training uses progressive epochs to find optimal tree count. Conservative hyperparameters allow more trees without overfitting.

## What Was Removed (Breaking Changes)

### Removed from Previous System

1. Phase training (grid search + fine-tuning iterations)
2. SL/TP optimization during training
3. Composite scoring for SL/TP selection
4. Generation versioning (GEN 6/7/8/9)
5. Hour-0 feature auto-detection
6. GEN 9 experimental features (S/D zones, microstructure)
7. CLI flags: `--legacy`, `--gen9`, `--target`, `--max-iter`, `--force`
8. Hyperparameter iteration scripts (train_iterate.py, train_progressive.py)

### Why These Were Removed

- **Phase training:** Caused overfitting to specific SL/TP values
- **SL/TP optimization:** Introduced optimization degradation, anchoring to local minima
- **Generation versioning:** Added complexity, made model behavior unpredictable
- **Hour-0 features:** Required extra data fetching, added marginal value
- **GEN 9 features:** Too complex, not enough improvement to justify
- **Iteration scripts:** Added complexity, hyperparameters should be manually tuned via config.py

## Training Process

### Old System (Problematic)

```
Phase 0: Data prep
Phase 1: Grid search 16 SL/TP combinations
Phase 2: Fine-tune around best with random perturbations
Result: Overfitted to training period, regressed on new data
```

### New System (Progressive - Like YOLO)

```
Before epochs (runs ONCE):
  1. Create single MLBacktest object
  2. This object will cache pooled data after first use

Epoch 1 (slower - ~60s for pooling + training):
  1. Load and pool data from all tickers (4 years training + 1 year validation)
  2. Pre-calculate features
  3. Cache pooled data in MLBacktest object
  4. Noise downsample + magnitude weight
  5. Train XGBoost with N trees (10 trees for epoch 1)
  6. Evaluate on validation period
  7. Display results

Epochs 2-10 (fast - ~1-2s each):
  1. Reuse cached pooled data (no re-pooling!)
  2. Noise downsample + magnitude weight (randomized each epoch)
  3. Train XGBoost with N trees (20, 30, 40... 100 trees)
  4. Evaluate on validation period
  5. Track if best model so far
  6. Display live results table

After all epochs:
  - Save best performing model
  - Show which epoch performed best
```

**Key optimization:** Data pooling happens ONCE by reusing the same MLBacktest object across all epochs. This saves ~45 seconds per epoch (what was previously wasted on re-pooling).

**Progressive improvement across epochs, like training YOLO.**

**Example: 100 trees over 10 epochs**

- Epoch 1: 10 trees
- Epoch 2: 20 trees
- Epoch 3: 30 trees
- ...
- Epoch 10: 100 trees (final)

Each epoch you see the model get better as more trees are added. Best model is automatically saved.

## File Structure

### Core Files

- `config.py` - All configuration (training, exit, portfolio)
- `ml/features.py` - Universal feature engineering (51 features)
- `ml/model.py` - XGBoost model wrapper
- `scripts/train.py` - Simplified training script
- `scripts/eval.py` - Backtesting engine
- `scripts/morning_signals.py` - Daily signal generation
- `scripts/eod_retrain.py` - End-of-day processing

### Model Storage

- `models/global_xgb_champion.pkl` - Active model binary
- `models/global_xgb_champion.json` - XGBoost native format
- `models/champion_metadata.json` - Performance metrics (display only, not used for SL/TP)
- `models/training_session_{timestamp}.json` - Training logs

### Data Storage

- `data/ihsg_trading.db` - SQLite database with price history
- `.cache/` - Temporary cached data (can be deleted)

### Data Quality Filters

**Updated liquidity requirements (config.portfolio):**

- `min_avg_volume: 2M shares` (was 1M) - More liquid stocks only
- `min_market_cap: 2T IDR` (was 1T) - Larger, more stable companies
- `min_data_points: 252` (was 100) - Require 1 year of history minimum

**Purpose:** Focus on high-quality, liquid stocks to reduce slippage and improve fill rates

## Feature Set Details

**Total: 56 features** (51 base + 5 intraday behavior proxies)

### Base 51 Features

**Price & Returns:**

- Returns: 1d, 2d, 3d, 5d, 10d, 20d, log_return
- Moving Averages: SMA 20/50, price_to_ma10/20/50
- Technical: RSI, MACD (line/signal/hist), ATR
- Volatility: 5d/20d vol, vol_ratio, vol_zscore

**Volume Indicators (NEW - Added 2024-12-30):**

- Basic: volume_sma, rel_volume, volume_change, relative_volume
- **Advanced:** MFI (Money Flow Index), CMF (Chaikin Money Flow), ADL ROC (Accumulation/Distribution)
- **Derived:** OBV trend, volume-price alignment

**Price Patterns:**

- Intraday: hl_range, gap, close_position, intraday_range_pct
- Momentum: mean_reversion_strength

**Lagged Features:**

- Returns: return_lag_1/2/3/5 (added lag_10/15/20 to feature_lags config)
- Indicators: rsi_lag1, rsi_change

**Calendar:**

- is_month_start/end, dow_0/1/2/3/4 (one-hot day of week)

### Intraday Behavior Proxies (5 features)

- `upper_shadow_ratio` - Detects rejection at highs (upper wick / body)
- `gap_size` - Gap magnitude (overnight gap)
- `gap_followthrough` - Close vs open after gap (gap strength)
- `intraday_momentum` - Close position in daily range (strength)
- `fade_signal` - Combined metric for gap exhaustion

## Signal Thresholds

**Universal across all scripts (from config.exit.signal_threshold):**

- `signal_threshold = 0.55` - Selective for 85%+ win rate (current)
- Focus on quality over quantity
- Expect 10-15 trades per month

**Training vs Production:**

- **Training:** Uses threshold from config (0.55 current)
- **Production:** Uses config.exit.signal_threshold (synchronized)
- **Morning Signals:** ML threshold = 0.55, Market orders = 0.85 (threshold + 0.30)

**Threshold guide:**

- 0.40 = ~70% win rate, 20-30 trades/month (aggressive)
- 0.50 = ~80% win rate, 10-20 trades/month (balanced)
- 0.55 = ~85% win rate, 10-15 trades/month (current - selective)
- 0.60 = ~90% win rate, 5-10 trades/month (very selective)

## Trading Logic

### Entry Rules

1. Pass screener filters (price > 50, volume > 2M, market cap > 2T IDR)
2. ML score > 0.55 (selective threshold - configurable in config.py)
3. Technical buy signal confirmed
4. Max 8 positions (was 10), max sector exposure 25%
5. Position size: 12.5% per trade (was 10%)
6. No IHSG crash detected (via RegimeDetector)

### Exit Rules (Priority Order)

1. Stop loss hit: entry × (1 - 2.0 × ATR%) - Tighter stops
2. Take profit hit: entry × (1 + 5.0 × ATR%) - Realistic targets
3. Trailing stop: 2.5× ATR (let winners run)
4. Time stop: 5 days maximum hold
5. Fixed fallback: 8% max loss, 15% min profit
6. IHSG crash detected: emergency exit

### Position Sizing (Signal-Strength-Based)

**Dynamic ATR-Based Volatility Targeting:**

```python
# Base calculation: Risk Budget / (ATR × Multiplier)
risk_budget = portfolio_value * 0.01  # 1% risk per trade
risk_per_share = atr * 2.0           # ATR multiplier for stop distance
base_shares = risk_budget / risk_per_share

# Signal strength adjustment (0.70-1.00 range)
confidence_mult = 0.5 + (ml_score * 0.5)  # 0.85x to 1.00x
final_shares = base_shares * confidence_mult

# Example with 100M portfolio, 1000 IDR entry, 50 IDR ATR:
# - Signal 0.70 → 8.5% position (85k shares)
# - Signal 0.85 → 9.25% position (92.5k shares)
# - Signal 1.00 → 10% position (100k shares)
```

**Implementation:** Uses `strategy/position_sizer.py` with ATR-based volatility targeting as primary method.

**Key Features:**

- Higher conviction trades get larger positions
- Marginal signals (0.70-0.75) get smaller allocations
- Automatic risk control via max position constraints
- Regime-aware adjustments (reduce in high volatility)

## Common Operations

### Train New Model

```bash
# Progressive training (default: 10 epochs)
uv run python scripts/train.py --days max --train-window max

# With custom configuration
uv run python scripts/train.py --days max --train-window max --max-depth 7 --n-estimators 200 --epochs 10

# Quick training (fewer epochs)
uv run python scripts/train.py --days max --train-window max --epochs 5

# Thorough training (more epochs for fine control)
uv run python scripts/train.py --days max --train-window max --epochs 20
```

**What you'll see:**

```
Starting Progressive Training (10 epochs)

Creating backtester (data will be cached after first epoch)

Epoch 1/10 - Training with 10 trees
  Pooling training data from 957 tickers... (60.5s)
  Win Rate: 68.2% | W/L: 2.1x | Trades: 44 | Return: 12.3% | Status: NEW BEST

Epoch 2/10 - Training with 20 trees
  Using cached pooled training data
  Win Rate: 72.5% | W/L: 2.3x | Trades: 47 | Return: 15.1% | Status: NEW BEST

Epoch 3/10 - Training with 30 trees
  Using cached pooled training data
  Win Rate: 75.8% | W/L: 2.5x | Trades: 51 | Return: 18.2% | Status: NEW BEST
...

Training Complete!
Best model: Epoch 8 with 82.4% win rate
```

**Performance:**

- Epoch 1: ~60-75s (pooling 4 years data + training)
- Epoch 2+: ~1-2s each (cached data, training only)
- Total 10 epochs: ~75s (vs ~600s if re-pooling each time)

### Backtest

```bash
uv run python scripts/eval.py --start 2024-01-01 --end 2025-12-30
```

### Generate Signals

```bash
uv run python scripts/morning_signals.py
```

### EOD Processing

```bash
uv run python scripts/eod_retrain.py
```

## Important Implementation Details

### Walk-Forward Training (eval.py)

- For each trading day, trains on previous `train_window` days (1008 default = 4 years)
- Pools data from all tickers (up to 252 days per ticker)
- Each ticker contributes tail(252, len(data)) rows
- Batch prediction for efficiency

### Noise Downsampling (eval.py:301-321)

```python
noise_mask = (ret_combined.abs() < 0.005)  # Days with <0.5% price change
keep_noise_indices = random.choice(noise_indices, size=len(noise_indices) // 2)
```

Purpose: Prevent model from learning "no movement" as default prediction

### Magnitude Weighting (eval.py:323-326)

```python
sample_weights = 1.0 + (ret_combined.abs() * 10.0)
```

Purpose: Days with larger price movements get higher weight during training

### IHSG Crash Filter (eval.py:594-601)

```python
if regime_detector.is_market_crash(ihsg_data, current_date):
    skip_entry = True
```

Purpose: Avoid new entries during market-wide selloffs

## Metadata Structure

`models/champion_metadata.json`:

```json
{
  "xgboost": {
    "win_rate": 0.82,
    "wl_ratio": 3.45,
    "total_trades": 156,
    "total_return": 18.7,
    "sharpe_ratio": 1.89,
    "sl_atr_mult": 2.5,
    "tp_atr_mult": 4.0,
    "signal_threshold": 0.5,
    "date": "2025-12-30 10:30:00",
    "feature_count": 51,
    "max_depth": 5,
    "n_estimators": 80,
    "best_epoch": 8
  }
}
```

**Note:** SL/TP in metadata is for display only. Actual values come from config.py. The `total_return` value is already in percentage form (18.7 = 18.7%).

## Position Sizing Modules

The system includes two position sizing implementations in `strategy/`:

### 1. position_sizer.py (Currently Used)

**Multi-factor ATR-based volatility targeting with signal-strength weighting**

**Primary Method:**

```python
# ATR-based volatility targeting
risk_budget = portfolio_value * 0.01  # 1% risk per trade
risk_per_share = atr * 2.0
base_shares = risk_budget / risk_per_share

# Signal strength adjustment
confidence_mult = 0.5 + (ml_score * 0.5)  # 0.85x to 1.00x
final_shares = base_shares * confidence_mult * regime_mult
```

**Features:**

- ATR-based primary sizing (use_atr_targeting=True)
- Confidence multiplier based on ML score
- Regime adjustments (CRASH=0x, HIGH_VOL=0.7x, LOW_VOL=1.2x)
- Maximum position constraints (default 10% per position)
- Legacy multi-factor fallback (Kelly + risk-based + volatility)

**Used in:** `scripts/eval.py` line 669-680

### 2. kelly_sizer.py (Available but Not Used)

**Pure Kelly Criterion implementation**

**Formula:**

```python
# Kelly formula: f* = (p × b - q) / b
kelly_fraction = (win_prob * wl_ratio - loss_prob) / wl_ratio
position_size = kelly_fraction * kelly_fraction_multiplier  # 0.25 = quarter-Kelly
```

**Features:**

- Fractional Kelly (25% default for safety)
- Signal-strength adjustments to win probability
- Historical edge tracking (recent 50 trades)
- Drawdown protection (reduce exposure after 3+ losses)
- Position range: 2-15% (configurable)

**Differences from position_sizer.py:**

- Simpler, more aggressive sizing
- Pure mathematical Kelly approach
- Less sophisticated than position_sizer's ATR targeting
- Better for experienced traders comfortable with Kelly

**To use:** Replace position_sizer with kelly_sizer in eval.py imports

## Blacklist Protection

### Blacklist Integration

The system maintains a blacklist of suspended/illiquid stocks in `data/blacklist.py` (currently 69 tickers).

**Three-layer protection:**

1. **clean_universe.py** - Filters blacklist BEFORE checking activity (O(1) set lookup)
2. **morning_signals.py** - Filters during signal generation AND warns on existing positions
3. **Data validation** - Prevents training on problematic stocks

**Example blacklisted tickers:** ALMI.JK, ARMY.JK, WIKA.JK, WSKT.JK (suspended/delisted)

**Implementation:**

```python
from data.blacklist import BLACKLIST_UNIVERSE

# Filter universe
active_stocks = [t for t in universe if t not in BLACKLIST_UNIVERSE]

# Check existing positions
if ticker in BLACKLIST_UNIVERSE:
    log(f"WARNING: {ticker} is BLACKLISTED - exit recommended!")
```

## Known Issues and Limitations

### Current Limitations

1. Single global model (not sector-specific)
2. No intraday trading (daily signals only)
3. No correlation management between positions
4. ~~No dynamic position sizing based on volatility~~ ✅ **IMPLEMENTED** (signal-strength-based)

### Why These Are Not Implemented

- **Simplicity:** Each adds complexity and potential failure modes
- **Effectiveness:** Current approach works well enough
- **Maintenance:** Simpler system is easier to debug and improve

### Recently Implemented

- ✅ **Signal-strength-based position sizing** (Dec 2025)
  - Integrated `strategy/position_sizer.py` into eval.py
  - ATR-based volatility targeting with confidence multiplier
  - Higher conviction trades get larger positions
  - Risk-controlled (1% risk per trade, max 10% per position)

### Future Considerations

- Sector-specific models (if global model plateaus)
- Correlation filtering (avoid 3+ highly correlated positions)
- Adaptive threshold adjustment based on market regime

## Testing and Validation

### Before Production

1. Syntax check all Python files
2. Train fresh model on recent data
3. Backtest on out-of-sample period
4. Verify signal generation produces reasonable output
5. Paper trade for 2-4 weeks

### Monitoring

1. Track daily win rate (should be 70%+)
2. Monitor drawdown (alert if > 15%)
3. Check signal count (should have 5-10 daily)
4. Verify position exits are triggering correctly

## Code Style Guidelines

### Enforced Standards

1. No emoji in code or comments (except user-facing CLI)
2. Docstrings for all public functions
3. Type hints for function parameters
4. No magic numbers - use config.py
5. Descriptive variable names (no single letters except i, j, k in loops)

### File Organization

- Config: `config.py` only
- Features: `ml/features.py` only
- Training: `scripts/train.py` only
- No duplicate logic across files

## Hyperparameter Tuning

**Manual tuning process:**

1. Edit `config.py` to change `max_depth` or `n_estimators`
2. Run `train.py` with new config
3. Compare results (win rate, sharpe, drawdown)
4. Keep best configuration in config.py

**Recommended values to try:**

- `max_depth`: 3 (simple), 5 (default), 7 (complex), 10 (very complex)
- `n_estimators`: 50 (fast), 100 (default), 150, 200 (slow but thorough)

**Signs of overfitting:**

- Training win rate > 95% but validation < 80%
- Backtest performance much worse than training
- Solution: Reduce `max_depth` or `n_estimators`

**Signs of underfitting:**

- Low win rate in both training and validation
- Solution: Increase `max_depth` or `n_estimators`

## Troubleshooting

### Model Performance Degraded

**Symptom:** Win rate drops below target
**Solution:** Retrain with fresh data, check for data quality issues, verify hyperparameters in config.py

### Too Few Signals

**Symptom:** < 3 signals per day
**Solution:** Check screener filters, verify data freshness, lower signal_threshold

### Too Many Signals

**Symptom:** > 15 signals per day
**Solution:** Raise signal_threshold, tighten screener filters

### Training Fails

**Symptom:** Error during train.py
**Solution:** Check data availability, verify 3 years of history exists, clear cache

### Training Too Slow (Each Epoch Takes 45s+)

**Symptom:** Every epoch pools training data, taking 40-50 seconds each
**Root Cause:** Creating new MLBacktest object inside epoch loop
**Solution:** Create ONE MLBacktest object before the loop, reuse it across epochs
**Example:**

```python
# WRONG - creates new object each epoch
for epoch in range(1, epochs + 1):
    bt = MLBacktest(...)  # Cache lost!
    results = bt.run(...)

# CORRECT - reuse same object
bt = MLBacktest(...)  # Create once
for epoch in range(1, epochs + 1):
    config.ml.n_estimators = current_estimators
    results = bt.run(...)  # Cached data reused
```

### Predictions All Same

**Symptom:** All stocks get same ML score
**Solution:** Retrain model, check feature engineering, verify data variance

### Percentage Display Errors

**Symptom:** Returns showing as 75.8% instead of 0.8%
**Root Cause:** Mixing `.1%` and `.1f%` format specifiers
**Rule:**

- Use `.1%` for ratio values (0.0-1.0, like win_rate from metadata)
- Use `.1f%` for already-percentage values (like total_return from eval.py)
  **Example:** `eval.py` returns `total_return * 100`, so display with `f"{total_return:.1f}%"` not `f"{total_return:.1%}"`

## Dependencies

### Core Libraries

- Python 3.9+
- XGBoost 2.0+
- pandas, numpy
- scikit-learn
- yfinance (data fetching)
- rich (CLI formatting)

### Package Management

- Uses `uv` for dependency management
- All deps in `pyproject.toml`
- Run `uv sync` to install

## Git Strategy

### Branches

- `main` - Stable production code
- No feature branches (direct commits to main)

### Commit Messages

- Format: `{type}: {description}`
- Types: feat, fix, refactor, docs, test
- Keep commits atomic and focused

### What to Commit

- Source code
- Configuration files
- Documentation
- Data schemas

### What NOT to Commit

- Models (_.pkl, _.json in models/)
- Data (\*.db in data/)
- Cache (.cache/)
- Training logs
- Environment files

## Performance Benchmarks

### Current Targets (Conservative Balanced Strategy - 0.55 threshold)

- Win Rate: 85%+ (selective 0.55 threshold)
- Average PnL per Trade: 15% (realistic 1-week target)
- W/L Ratio: 2.5-3.0x (balanced 2:5 SL/TP ratio)
- Sharpe Ratio: 2.0-2.5+ (high win rate + consistent returns)
- Max Drawdown: < 8% (tighter risk management)
- Total Trades: 10-15 per month (selective quality)
- SL/TP: 2.0x / 5.0x ATR (realistic stops & targets)
- Max Holding: 5 days (short-term swing trading)
- Training Data: 4 years (1008 trading days)
- Position Size: 12.5% per trade (8 max positions)
- XGBoost: 500 shallow trees @ 0.02 learning rate (very conservative)

### Alternative Strategies

**Ultra-Selective (0.70 threshold):**

- Win Rate: 90-95%, Trades: 5-10/month, Very conservative

**Moderate (0.40 threshold):**

- Win Rate: 70-80%, Trades: 20-30/month, More aggressive

### Legacy Metrics (Moderate Strategy - 0.40 threshold)

- Win Rate: 70-80%
- W/L Ratio: 1.8-2.5x
- Total Trades: 20-30 per month
- SL/TP: 2.0x / 3.0x ATR

### Training Performance

- Progressive training (10 epochs): ~75s total (Epoch 1: 60s, Epochs 2-10: 1-2s each)
- Single epoch training: 75-105 seconds (4 years data, more than 3 years)
- Feature engineering: 10-15 seconds per ticker (parallelized)
- Prediction time: < 1 second for 956 tickers

## Integration Points

### Data Sources

- Yahoo Finance (primary)
- IDX directly (future consideration)

### External Systems

- None currently
- Future: Broker API for automated execution

## Security Considerations

### Sensitive Data

- No credentials stored in code
- Database is local (no network access)
- No API keys required (yfinance is free)

### Risk Management

- No automated execution (manual review required)
- Position size limits enforced
- Stop losses always set
- Max drawdown monitoring

## Version History

### v2.1.0 (2025-12-30) - Signal-Strength Position Sizing & Swing Trading

- **Signal-strength-based position sizing** integrated into backtesting
  - Uses `strategy/position_sizer.py` with ATR-based volatility targeting
  - Higher conviction trades (0.95+) get larger positions (~10-12%)
  - Marginal trades (0.70-0.75) get smaller positions (~6-8%)
  - Confidence multiplier: 0.5 + (signal_score × 0.5)
- **Conservative balanced configuration** for sustainable returns
  - SL/TP: 2.0x/5.0x ATR (realistic stops & targets)
  - Max holding: 5 days (short-term swing trades)
  - Signal threshold: 0.55 (selective for 85%+ win rate, 10-15 trades/month)
  - Target: 15% average PnL per trade (realistic 1-week target)
  - Position sizing: 12.5% per trade (8 max positions, was 10% / 10 positions)
- **Blacklist protection** integrated
  - 69 suspended/illiquid stocks filtered in clean_universe.py
  - morning_signals.py warns on blacklisted positions
- **Training improvements & XGBoost tuning**
  - Training window: 1008 days (4 years - maximum historical data)
  - Very strong regularization (gamma=0.2, reg_alpha=1.0, reg_lambda=2.0)
  - Conservative hyperparameters: 500 trees @ 0.02 LR, max_depth=4, min_child_weight=5
  - Subsample/colsample: 0.7 (was 0.8) for more diversity
  - Composite score for model selection: WR × Return × √Sharpe × log(1 + trades/100)
  - Fixed hardcoded 0.40 threshold bugs in train.py and eval.py
  - Added 56 features (was 51): MFI, CMF, ADL, OBV trend, volume-price alignment
- **Bug fixes**
  - JSON serialization error (numpy bool → Python bool)
  - Session folder structure for epoch models
  - Threshold synchronization across all scripts

### v2.0.1 (2025-12-30) - Performance Optimization

- Fixed data pooling cache bug (was re-pooling every epoch)
- Fixed return calculation display bug (75.8% → 0.8%)
- Optimized progressive training (60s total for 10 epochs)

### v2.0 (2025-12-30) - Simplification Refactor

- Removed phase training
- Fixed SL/TP from config
- Universal 51-feature set
- Single training script (train.py only)
- Removed generation versioning
- Progressive training like YOLO

### v1.x (Legacy)

- Phase training with grid search
- SL/TP optimization
- Generation versioning (GEN 6/7/8/9)
- 4-year training window
- Complex composite scoring

## Contact and Support

### Documentation

- README.md - User guide
- presentation.md - Technical deep dive
- project-knowledge.md - This file (developer reference)

### Issue Tracking

- GitHub Issues (if using GitHub)
- Local notes (if not)

---

**Remember:** When in doubt, keep it simple. The goal is a stable, predictable system that makes money consistently, not a complex machine learning showcase.
