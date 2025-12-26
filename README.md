# Paperium: High-Performance IHSG ML Trading Bot

Paperium is an automated trading system optimized for the Indonesia Stock Exchange (IHSG). It leverages a multi-model ensemble of XGBoost and Supply/Demand price action zones to achieve high-precision signals with target win rates exceeding 80%.

## Key Features

*   **Universal Data Pooling**: Trains on the entire liquid IHSG universe (IDX80, Kompas100), generalizing across 460+ stocks.
*   **High-Potential Screener**: Pre-filters stocks for trend (EMA 200), liquidity, and ATR-based volatility before ML processing.
*   **Auto-Training Optimization**: A feedback loop that tunes model parameters to achieve >80% monthly win rates.
*   **Interactive Morning Signals**: Professional-grade CLI to review recommendations and execute trades in **Live** or **Test** mode.
*   **Champion vs. Challenger EOD**: End-of-day retraining that safely upgrades the "Champion" model only if a "Challenger" performs better.

## Installation

Ensuring you have `uv` installed, then run:

```bash
# Clone the repository
cd paperium

# Install dependencies
uv sync
```

## 7-Step Automated Workflow

The system is designed to run autonomously or under user supervision. Use the following scripts in order:

### 1. Universe Maintenance
Keep your stock list clean and up-to-date.
```bash
uv run python scripts/clean_universe.py  # Removes inactive stocks
uv run python scripts/sync_data.py       # Fetches 2 years of history for all tickers
```

### 2. Model Optimization
Train the brain. This loop iterates until it finds a model meeting the 80% WR target.
```bash
uv run python scripts/auto_train.py
```
*   *Champion model is saved to `models/global_xgb_champion.pkl`.*

### 3. Morning Signals (Daily Strategy)
Run this before the market opens (08:30–08:50 WIB).
```bash
uv run python scripts/morning_signals.py
```
*   **Test Mode**: Displays top-5 candidates and signals without changing your portfolio.
*   **Live Mode**: Commits trades to the database and tracks your positions.

### 4. EOD Retraining (Post-Market)
Run after the market close (16:00+ WIB).
```bash
uv run python scripts/eod_retrain.py
```
*Updates existing positions (SL/TP hits) and retrains the model with today's data.*

Paperium operates with a modular dual-brain architecture, enabling independent strategy management:

### Core Models
*   **XGBoost (Champion)**: High-performance gradient boosted trees focused on next-day return probability.
*   **GD/SD (Alternative)**: Structural strategy combining Gradient Descent price tracking with Supply/Demand zone detection.

## Execution Guide
For the easiest experience, use the unified runner:
```bash
python run.py
```
This interactive menu handles all workflows:
- **Morning Ritual**: Generate and review signals.
- **Evening Update**: Retrain and update positions.
- **Model Lab**: Targeted training for specific strategies.

### Quality Control: Champion vs Challenger
The system ensures peak performance through a rigorous validation loop:
1.  **Persistence**: The best-known metrics for each strategy are stored in `models/champion_metadata.json`.
2.  **Replacement Policy**: New models only replace the "Champion" if they demonstrate a strictly superior Win Rate during training/backtesting.
3.  **Modular Training**: 
    - Full sweep: `uv run python scripts/auto_train.py`
    - Targeted: `uv run python scripts/train_model.py --type <xgboost|gd_sd> --target 0.85`
4.  **Daily Refinement**: `scripts/eod_retrain.py` automatically attempts to improve the champions every day after market close.

## Configuration

Modify `config.py` to adjust:
- `MLConfig`: Training windows and min sample requirements.
- `PortfolioConfig`: Position sizing and maximum exposure.
- `ExitConfig`: Default Stop-Loss and Take-Profit percentages.

## Directory Structure
- `/models`: Persisted Champion models.
- `/data`: SQLite database (`trading.db`) and historical cache.
- `/scripts`: Main entry points for the automation workflow.
- `/ml`: Model architectures (XGBoost, SupplyDemand, Ensemble).

## Backtest Results
The latest baseline using the **XGBoost Global Champion** achieved:
- **Win Rate**: 89.7%
- **Total Return (90 days)**: +156.2%
- **Max Drawdown**: -1.0%

---
*Disclaimer: Trading stocks involves risk. This software is for paper trading and educational purposes.*
