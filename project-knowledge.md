# Paperium Project Knowledge

**System Type**: Deep Learning Quantitative Trading
**Target Market**: Indonesia Stock Exchange (IHSG)
**Model Architecture**: LSTM (Long Short-Term Memory)
**Labeling Scheme**: Triple Barrier Method

## 1. Core Philosophy

Paperium has transitioned from traditional ML (XGBoost + Technical Indicators) to **Deep Learning on Raw Data**. The hypothesis is that neural networks can learn better feature representations from raw price sequences than human-engineered indicators (RSI, MACD, etc.).

## 2. The Model (LSTM)

We use a standard LSTM architecture designed for time-series classification.

- **Input**: `(Batch, Sequence_Length, Features)`
  - Sequence Length: **100 days** (Optimized for 5-day horizon)
  - Features: **5** (Open, High, Low, Close, Volume - Normalized)
- **Hidden Layers**:
  - Layer 1: LSTM (Input -> Hidden Size 8)
  - Layer 2: LSTM (Hidden Size 8 -> Hidden Size 8)
- **Output**: Fully Connected Layer -> 3 Classes (Softmax)
  - Class 0: **LOSS** (Hit lower barrier first)
  - Class 1: **NEUTRAL** (Time expired before any barrier)
  - Class 2: **PROFIT** (Hit upper barrier first)

## 3. Triple Barrier Labeling (TBL)

Instead of fixed "Close-to-Close" returns, we use TBL to capture the path dependency of trading.

- **Horizon**: **5 Days**. This is the maximum holding period.
- **Barrier Width**: **3.0%**.
  - If High > Entry \* 1.03 first -> Label 2 (Profit)
  - If Low < Entry \* 0.97 first -> Label 0 (Loss)
  - If neither happens by Day 5 -> Label 1 (Neutral)

_Optimization_: We found 3% / 5 days to be the sweet spot for IHSG volatility.

## 4. Data Pipeline

1.  **Ingestion**: `yfinance` fetches daily OHLCV.
2.  **Normalization**:
    - Prices are normalized relative to the _first day_ of the 100-day window (`p_t / p_0 - 1`).
    - Volume is log-normalized.
3.  **Sequence Generation**:
    - Rolling window of size 100.
    - stride = 1 (creates overlapping sequences).
4.  **Splitting**:
    - Time-ordered split (not random shuffle) to prevent look-ahead bias.
    - Training: First 80% (chronological).
    - Validation: Last 20% (chronological).

## 5. Components Status

| Component            | Status        | Notes                                             |
| :------------------- | :------------ | :------------------------------------------------ |
| **Data Fetcher**     | ✅ Stable     | SQLite backend with hourly caching.               |
| **Screener**         | ✅ Simplified | Blacklist filtering (72 illiquid stocks).         |
| **Feature Eng.**     | ✅ Replaced   | Generates sequences + TBL labels.                 |
| **Model**            | ✅ PyTorch    | Saved as `best_lstm.pt`.                          |
| **Training**         | ✅ Active     | `train.py` handles loop & early stopping.         |
| **Evaluator**        | ✅ Active     | `eval.py` runs walk-forward backtest.             |
| **Signal Generator** | ✅ Active     | `signals.py` with confidence-weighted allocation. |

## 6. Signal Generation & Capital Allocation

The signal generation system in `signals.py` provides flexible capital allocation:

- **Blacklist Filtering**: Automatically excludes 72 illiquid/suspended stocks from `data/blacklist.py`.
- **Confidence Weighting**: When allocating capital, higher confidence signals receive proportionally larger allocations.
- **Formula**: `allocation_i = total_capital × (confidence_i / sum_of_confidences)`
- **Latest Data**: Optional `--fetch-latest` flag fetches current market data from Yahoo Finance.
- **Flexible Output**: Can show all signals or only allocated positions with P/L estimates.

## 7. Training Strategy

- **Loss Function**: Cross Entropy Loss.
- **Optimizer**: Adam (LR=0.001).
- **Batch Size**: 64.
- **Early Stopping**: Stops if Validation Loss doesn't improve for 10 epochs.
- **Device**: MPS (Mac) / CUDA / CPU auto-detection.

## 7. Known Issues / Future Work

- **Imbalanced Classes**: In low volatility, Class 1 (Neutral) dominates. Class weights might be needed.
- **Inference Speed**: LSTM inference is slower than XGBoost but acceptable (~5-10s for full universe).
- **Hyperparameter Tuning**: `tune_lstm.py` exists to refine hidden size/layers.
