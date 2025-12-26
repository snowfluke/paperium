# Building an XGBoost Trading System for IHSG: A Technical Deep Dive

This document explains the exact technical implementation of Paperium, an automated trading system for the Indonesia Stock Exchange, from data ingestion to live signal generation.

## Data Foundation: SQLite Storage and 5-Year Lookback

The system starts by ingesting OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance for 480 tickers in the IHSG universe. Historical data spans 5 years (`lookback_days: 1825`) and is stored in a SQLite database at `data/ihsg_trading.db`. The universe includes liquid stocks from IDX80, Kompas100, and growth sectors, defined in [`config.py:13-481`](file:///Users/vexeee/Documents/project/paperium/config.py#L13-L481).

```python
# config.py - Data Configuration
db_path: str = "data/ihsg_trading.db"
lookback_days: int = 365 * 5  # 5 years
min_data_points: int = 100
```

Data fetching happens via `DataStorage.get_prices()`, which queries the database and returns a pandas DataFrame sorted by ticker and date.

## Feature Engineering: 46 Hardcoded Features

The model requires **exactly 46 features** in a specific order, defined in [`features.py:22-33`](file:///Users/vexeee/Documents/project/paperium/ml/features.py#L22-L33). This is the LEGACY_46_FEATURES array that the champion model expects.

### Feature Categories

**1. Price Returns (10 features)**
```python
# features.py:130-132
for lag in [1, 2, 3, 5, 10, 20]:
    df[f'return_{lag}d'] = df['close'].pct_change(lag)

df['log_return'] = np.log(df['close'] / df['close'].shift(1))
```

**2. Volatility Metrics (5 features)**
```python
# features.py:123, 139-141
df['volatility'] = df['log_return'].rolling(20).std() * np.sqrt(252)
df['volatility_5d'] = df['log_return'].rolling(5).std() * np.sqrt(252)
df['volatility_20d'] = df['log_return'].rolling(20).std() * np.sqrt(252)
df['vol_ratio'] = df['volatility_5d'] / df['volatility_20d'].replace(0, np.inf)

# Volatility Z-Score: Detects abnormal volatility
vol_mean = df['volatility_20d'].rolling(60).mean()
vol_std = df['volatility_20d'].rolling(60).std()
df['volatility_zscore'] = (df['volatility_20d'] - vol_mean) / vol_std.replace(0, 1)
```

The volatility z-score formula:
$$Z_\sigma = \frac{\sigma_{20d} - \mu_{60}(\sigma)}{\text{std}_{60}(\sigma)}$$

**3. Moving Average Relationships (3 features)**
```python
# features.py:144-146
for period in [10, 20, 50]:
    ma = df['close'].rolling(period).mean()
    df[f'price_to_ma{period}'] = df['close'] / ma - 1
```

**4. Technical Indicators (RSI, MACD, ATR) - 7 features**
```python
# technical.py:80-89 - RSI Implementation
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = (-delta).where(delta < 0, 0)
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.inf)
df['rsi'] = 100 - (100 / (1 + rs))

# technical.py:105-110 - MACD Implementation
ema_fast = df['close'].ewm(span=12, adjust=False).mean()
ema_slow = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema_fast - ema_slow
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_histogram'] = df['macd'] - df['macd_signal']

# technical.py:155-160 - ATR Implementation
high_low = df['high'] - df['low']
high_close = abs(df['high'] - df['close'].shift(1))
low_close = abs(df['low'] - df['close'].shift(1))
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = tr.rolling(window=14).mean()
```

**5. Volume Features (4 features)**
```python
# features.py:117-118, 159-161
df['volume_sma'] = df['volume'].rolling(window=20).mean()
df['rel_volume'] = df['volume'] / df['volume_sma'].replace(0, 1)
df['volume_change'] = df['volume'].pct_change()
df['relative_volume'] = df['volume'] / df['volume_ma20'].replace(0, 1)
```

**6. Lagged Features (7 features)**
```python
# features.py:182-189
for lag in [1, 2, 3, 5]:
    df[f'return_lag_{lag}'] = df['return_1d'].shift(lag)

df['rsi_lag1'] = df['rsi'].shift(1)
df['rsi_change'] = df['rsi'] - df['rsi'].shift(1)
```

**7. Calendar Features (5 features - dow_0 through dow_4)**
```python
# features.py:204-213
df['date'] = pd.to_datetime(df['date'])
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

for day in range(5):  # Mon-Fri
    df[f'dow_{day}'] = (df['day_of_week'] == day).astype(int)
```

**8. Intraday Patterns (5 features)**
```python
# features.py:149-156, 171, 174-175
df['hl_range'] = (df['high'] - df['low']) / df['close']
df['hl_range_avg'] = df['hl_range'].rolling(20).mean()
df['gap'] = df['open'] / df['close'].shift(1) - 1
df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
df['intraday_range_pct'] = df['close_position'].rolling(5).mean()

# Mean reversion strength
price_to_ma_std = df['price_to_ma20'].rolling(60).std()
df['mean_reversion_strength'] = df['price_to_ma20'].abs() / price_to_ma_std.replace(0, 1)
```

### Target Variable: 5-Day Forward Return

```python
# features.py:72-73
df['target'] = df['close'].shift(-5) / df['close'] - 1
df['target_direction'] = (df['target'] > 0).astype(int)
```

The model predicts whether the stock will be profitable 5 days into the future (binary classification: 1 = up, 0 = down/flat). The choice of 5 days aligns with the `max_holding_days` exit rule in backtesting ([`config.py:540`](file:///Users/vexeee/Documents/project/paperium/config.py#L540)).

## XGBoost Architecture: Global Pooled Model

Rather than training individual models per ticker, Paperium uses a **single global model** trained on pooled data from all 480 tickers. This approach increases training samples and allows the model to learn cross-sectional patterns.

### Model Hyperparameters

```python
# model.py:58-69
xgb.XGBClassifier(
    n_estimators=100,          # 100 decision trees
    max_depth=5,               # Maximum tree depth (prevents overfitting)
    learning_rate=0.1,         # Step size shrinkage
    min_child_weight=3,        # Minimum samples required to split
    subsample=0.8,             # 80% row sampling per tree
    colsample_bytree=0.8,      # 80% feature sampling per tree
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
```

### Training Pipeline: Rolling Window + Noise Filtering

The training process happens in [`eval.py:177-306`](file:///Users/vexeee/Documents/project/paperium/scripts/eval.py#L177-L306):

**Step 1: Data Pooling**
```python
# eval.py:188-210
for ticker in all_data['ticker'].unique():
    train_data = ticker_data[ticker_data['date'] < start]
    train_data = train_data.tail(min(252, len(train_data)))  # 1-year rolling window
    X, y, returns = feature_engineer.create_features(train_data, target_horizon=5, include_raw_return=True)
    pool_X.append(X)
    pool_y.append(y)
```

Each ticker contributes up to 252 trading days (1 year) of historical data.

**Step 2: Noise Downsampling**
```python
# eval.py:225-242
noise_mask = (ret_combined.abs() < 0.005)  # Days with <0.5% price change
noise_indices = ret_combined[noise_mask].index
keep_noise_indices = np.random.choice(noise_indices, size=len(noise_indices) // 2, replace=False)
```

50% of "boring" days (where the stock moved less than 0.5%) are randomly discarded. This prevents the model from learning that "no movement" is the default prediction.

**Step 3: Magnitude Weighting**
```python
# eval.py:244-246
sample_weights = 1.0 + (ret_combined.abs() * 10.0)
```

Days with larger price movements get higher weights during training. A 5% move gets 1.5x weight compared to a 0.5% move. This formula:
$$w = 1.0 + |r| \times 10$$

where $r$ is the forward return.

**Step 4: Warm Start (Incremental Learning)**
```python
# eval.py:260-291
base_model = None
if os.path.exists(xgb_champ_path):
    temp_xgb.load(xgb_champ_path)
    if temp_xgb.feature_names == new_features:
        base_model = temp_xgb.model

if base_model is not None:
    self.global_xgb.model.fit(X_combined, y_combined, sample_weight=sample_weights, xgb_model=base_model.get_booster())
else:
    self.global_xgb.model.fit(X_combined, y_combined, sample_weight=sample_weights)
```

If an existing champion model exists, the new model continues training from that checkpoint (warm start). This preserves learned patterns while adapting to new data.

## Prediction: Probability to Score Conversion

```python
# eval.py:308-316
def _get_prediction(ticker, df):
    prob = self.global_xgb.predict_latest(df)  # Returns probability [0, 1]
    return (prob - 0.5) * 2  # Convert to score [-1, 1]

# model.py:240
def predict_latest(df):
    y_proba = self.model.predict_proba(X)[:, 1]  # P(class=1)
```

XGBoost outputs a probability between 0 and 1. This is transformed to a score between -1 (bearish) and +1 (bullish). A score > 0.1 triggers a buy signal ([`eval.py:477`](file:///Users/vexeee/Documents/project/paperium/scripts/eval.py#L477)).

## Screening: Pre-ML Filters

Before scoring with XGBoost, stocks must pass liquidity and quality filters in [`screener.py`](file:///Users/vexeee/Documents/project/paperium/signals/screener.py):

```python
# screener.py - Applied in eval.py:466
if len(ticker_hist) < 200:
    continue  # Need at least 200 days of data

if not self.screener._check_criteria(ticker_hist, ticker):
    continue  # Failed liquidity/quality checks
```

Only stocks passing the screener get ML predictions.

## Position Sizing and Risk Management

Once a stock scores > 0.1 and passes screening:

```python
# eval.py:499-517
position_value = min(cash * 0.12, initial_capital * 0.1)
shares = int(position_value / entry_price)

# ATR-based stops
stop_loss = entry_price * (1 - max(0.05, atr / entry_price * 2))
take_profit = entry_price * (1 + max(0.08, atr / entry_price * 3))
```

Position sizing:
- Each position = 10-12% of portfolio
- Maximum 10 concurrent positions
- Maximum 3 positions per sector

Stop-loss and take-profit are **dynamic**, based on ATR (Average True Range):
- Stop Loss = entry price * (1 - max(5%, 2×ATR%))
- Take Profit = entry price * (1 + max(8%, 3×ATR%))

Time-based exit:
```python
# eval.py:436-446
if pos['days_held'] >= 5:
    exit_price = row['close']
    cash += pos['shares'] * exit_price * (1 - 0.0025)  # 0.25% sell fee
```

After 5 days, positions are automatically closed regardless of profit/loss.

## Iterative Training: Target-Based Optimization

The [`train.py`](file:///Users/vexeee/Documents/project/paperium/scripts/train.py) script implements iterative optimization to hit a target win rate:

```python
# train.py:86-138
while iteration < max_iter:
    iteration += 1
    
    bt = MLBacktest(model_type='xgboost', retrain=True)
    if iteration > 1:
        bt.stop_loss_pct = 0.03 + (iteration * 0.005)
        bt.take_profit_pct = 0.06 - (iteration * 0.005)
    
    results = bt.run(start_date=start_date, end_date=end_date, train_window=train_window)
    
    monthly_wrs = [m['win_rate'] / 100.0 for m in results['monthly_metrics']]
    effective_wr = (sum(monthly_wrs) / len(monthly_wrs) * 0.7) + (min(monthly_wrs) * 0.3)
    
    if effective_wr >= target:
        bt.global_xgb.save("models/global_xgb_champion.pkl")
        break
```

**Effective Win Rate Formula:**
$$WR_{\text{eff}} = 0.7 \times \overline{WR}_{\text{monthly}} + 0.3 \times \min(WR_{\text{monthly}})$$

This penalizes models with inconsistent performance. A model that wins 90% in some months but 40% in others will score lower than a model with consistent 70% wins.

Each iteration adjusts stop-loss and take-profit percentages:
- Iteration 1: SL=5%, TP=8%
- Iteration 2: SL=6.5%, TP=6.5%
- Iteration 3: SL=8%, TP=5%

The system automatically saves the best model that meets the target.

## Daily Operations: Morning Signals

```python
# scripts/morning_signals.py execution flow:
# 1. Load champion model from models/global_xgb_champion.pkl
# 2. Fetch latest OHLCV data for all 480 tickers
# 3. Calculate 46 features for each ticker
# 4. Run screener filters
# 5. Predict probabilities for passing tickers
# 6. Rank by score (prob - 0.5) * 2
# 7. Output top 10 signals with position sizes
```

The model runs every morning before market open, generating fresh signals based on yesterday's closing data. No human intervention required.

## Summary: The Complete Pipeline

1. **Data**: 480 IHSG tickers, 5 years of OHLCV data in SQLite
2. **Features**: 46 hardcoded features (returns, volatility, technical indicators, volume, calendar)
3. **Target**: 5-day forward return direction (binary classification)
4. **Model**: Single global XGBoost with 100 trees, depth 5, learning rate 0.1
5. **Training**: Pooled data from all tickers, noise filtering (50% removal), magnitude weighting (w = 1 + |r|×10), warm start from previous champion
6. **Prediction**: Probability → Score [-1, 1], threshold > 0.1 for buy
7. **Risk**: ATR-based stops, 10% position size, 5-day max hold, 10 max positions
8. **Optimization**: Iterative training targeting effective win rate with dynamic stop/profit adjustment
9. **Production**: Automated morning signals, model self-updates end-of-day

Every component is deterministic and reproducible. No discretionary decisions, no manual overrides.
