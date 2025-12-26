# Paperium: Quantitative Trading with XGBoost

This presentation explains the architecture and logic of Paperium, a quantitative trading system specifically designed for the Indonesia Stock Exchange (IHSG). It leverages Machine Learning to navigate market complexities and automate decision-making.

---

## 1. The Core Philosophy: Adaptive Intelligence

The financial market is a complex, non-linear system. Traditional static rules (e.g., "Buy when RSI < 30") often fail as market regimes change. 

Paperium shifts the paradigm from **Static Rules** to **Adaptive Learning**. Instead of following fixed formulas, the system "observes" the last 252 trading days (a full year) and learns which patterns preceded profitable moves.

---

## 2. The High-Level Process Flow

The system operates in a continuous feedback loop:

1. **Information Gathering**: Download historical OHLCV (Open, High, Low, Close, Volume) data for the IHSG universe.
2. **Knowledge Extraction (Feature Engineering)**: Convert raw prices into 46 distinct mathematical signals.
3. **Reasoning (XGBoost)**: The model processes these signals to estimate the probability of a positive return.
4. **Execution**: If the probability exceeds a threshold, the system ranks the opportunity and manages the entry/exit.
5. **Self-Refinement (EOD Retrain)**: Every evening, the system adds the new day's result to its memory and retrains if a better model is found.

---

## 3. Feature Engineering: The Language of the Model

The XGBoost model doesn't see "prices"; it sees "Features." Paperium uses a 46-feature set that captures different market dimensions:

### Momentum and Trend
We use standard indicators like SMA (Simple Moving Average), RSI (Relative Strength Index), and MACD (Moving Average Convergence Divergence).

### Mathematical Representations
To help the model understand price velocity and volatility, we use formal equations:

**Log Returns ($r_t$):**
Captures the percentage change in a way that is additive over time.
$$r_t = \ln\left(\frac{Price_t}{Price_{t-1}}\right)$$

**Volatility Z-Score ($Z_{\sigma}$):**
Determines if current volatility is "abnormal" compared to its historical mean ($\mu_{\sigma}$) and standard deviation ($\text{std}(\sigma)$).
$$Z_{\sigma} = \frac{\sigma - \mu_{\sigma}}{\text{std}(\sigma)}$$

**Relative Volume:**
Measures if current trading activity is higher than average, often indicating institutional interest.
$$RelVol = \frac{Volume_t}{SMA(Volume, 20)}$$

---

## 4. The Brain: XGBoost (Extreme Gradient Boosting)

### Why XGBoost?
XGBoost is a Tree-Based Ensemble model. Unlike simple linear models, it can:
- Capture non-linear relationships (e.g., "RSI is bullish only when Volume is high").
- Handle missing data and outliers automatically.
- Provide feature importance, showing exactly which indicators are driving the decisions.

### Binary Classification
The system solves a "True" or "False" problem:
- **Target**: Is the return after $N$ days $> 0$?
- **Output**: A probability score between 0 and 1.

---

## 5. Staying Relevant: Rolling Window Training

Markets evolve. A pattern that worked in 2022 might be obsolete in 2024. To combat this, Paperium uses a **Rolling Window** approach:

- The model is always trained on the most recent $W$ samples (e.g., 252 days).
- As each new day passes, the oldest data point is dropped, and the newest is added.
- **Daily Self-Refinement**: The system continuously tests a "Challenger" model against the current "Champion." Only the superior model is allowed to generate signals for the next day.

---

## 6. Execution and Risk Control

A high probability signal is not enough; survival requires risk management. Every trade is governed by:

1. **Position Sizing**: Allocating capital based on portfolio equity and risk per trade.
2. **Trailing Stop Loss**: A safety net that moves up as the price rises, protecting profits.
3. **Time Stop**: If a trade doesn't "work" within a specific timeframe (e.g., 5-10 days), the system exits to reallocate capital to fresher opportunities.

---

## 7. Conclusion

Paperium is not a "crystal ball." It is a disciplined, mathematical framework that removes human emotion from trading. Through the combination of rigorous Feature Engineering, the XGBoost algorithm, and a continuous learning loop, it seeks to maintain a quantitative edge in the Indonesian market.
