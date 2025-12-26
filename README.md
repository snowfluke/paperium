# Paperium: IHSG Quantitative Trading System

![Paperium Training Model](images/training-models.png)

Paperium is a high-performance quantitative trading system designed for the Indonesia Stock Exchange (IHSG). It leverages advanced machine learning (XGBoost) to predict short-term price movements and manage a modular trading portfolio.

## Key Architecture

Paperium uses a streamlined, high-performance architecture centered around a single, robust machine learning model:

-   **The Predictor (XGBoost):** A sophisticated Gradient Boosting model that analyzes dozens of technical indicators and historical price patterns to predict the probability of a positive return on the next trading day.

## Project Structure

-   `data/`: Data ingestion from Yahoo Finance and SQLite storage.
-   `ml/`: Core XGBoost model implementation and feature engineering.
-   `signals/`: Technical indicator generation and stock screening.
-   `strategy/`: Position management, exit logic (SL/TP), and portfolio sizing.
-   `scripts/`: Automation scripts for daily rituals and model optimization.
-   `models/`: Storage for trained champion models.

---

## Quick Start Guide

### 1. Model Training
Train the global XGBoost model using historical data:
```bash
# Targeted training (90 days eval)
uv run python scripts/train.py --days 90 --target 0.85

# Max window training (using 3 years of data)
uv run python scripts/train.py --days max --train-window max
```

### 2. Evaluation
Verify performance over a specific period:
```bash
uv run python scripts/eval.py --start 2024-01-01 --end 2025-09-30
```
### 3. Morning Ritual (Live Signals)
Generate trading recommendations before market open:
```bash
uv run python scripts/morning_signals.py
```

---

## Detailed Workflow

### Phase 1: Data Preparation
The `DataFetcher` retrieves the latest OHLCV data for the IHSG stock universe. Features are calculated via the `FeatureEngineer`, normalizing indicators like RSI, MACD, and volatility for the XGBoost model.

### Phase 2: Signal Generation
1.  **Screener:** Filters out illiquid or stable stocks based on volume and price volatility.
2.  **XGBoost Prediction:** The champion model generates a "confidence score" (0-100%) for each remaining stock.
3.  **Ranking:** Stocks are ranked based on their ML score and technical consensus.

### Phase 3: Position Management
The `PositionManager` executes trades based on the top-ranked signals. Each position is protected by:
-   **Trailing Stop Loss:** Dynamic SL based on ATR or percentage.
-   **Take Profit:** Fixed or dynamic TP targets.
-   **Time Stop:** Automatic exit after a set number of days to ensure capital velocity.

### Phase 4: Evening Self-Refinement
Every evening after market close, the `eod_retrain.py` script:
1.  Evaluates current positions and updates P&L.
2.  Incorporate today's price action into the training pool.
3.  Performs a "Champion Challenge": Trains a new model and only replaces the current one if it shows superior validation accuracy.

---

## Dashboard
Run the unified runner for an interactive CLI experience:
```bash
uv run python run.py
```

---

## Key Directories
- `/models`: Stores your `.pkl` Champion models.
- `/.cache`: Stores hourly price snapshots for faster execution.
- `/data`: Your SQLite database (`ihsg_trading.db`) containing price history.

---
*Disclaimer: Trading stocks involves significant risk. This bot is a tool for decision support. Always use Test Mode before committing to live trading.*
