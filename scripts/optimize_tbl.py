import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import TimedLogger

DB_PATH = "data/ihsg_trading.db"

def get_tickers():
    """Fetch all unique tickers from the database."""
    conn = sqlite3.connect(DB_PATH)
    tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM prices", conn)
    conn.close()
    return tickers['ticker'].tolist()

def load_data(ticker):
    """Load OHLC data for a specific ticker."""
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT date, open, high, low, close FROM prices WHERE ticker = '{ticker}' ORDER BY date"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df

def apply_triple_barrier(df, horizon, barrier):
    """
    Apply Triple Barrier Labeling.
    
    Paper Logic:
    - Labels generated using Low and High prices.
    - If High >= Entry * (1+TP) -> Label 2 (Buy/Profit)
    - If Low <= Entry * (1-SL) -> Label 0 (Sell/Loss)
    - If neither hit within T days -> Label 1 (Hold/Time Limit)
    - If both hit on same day -> Label 1 (Time Limit / No Move)
    
    Returns: label counts (0, 1, 2)
    """
    # Assuming barrier is symmetric for TP and SL as per paper optimization description (single %).
    # Paper optimized "percentage thresholds (7%-15%)". Implies TP=SL=Barrier.
    
    df = df.copy()
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    n = len(df)
    labels = np.zeros(n, dtype=int) + 1 # Default to 1 (Time Limit)
    
    # Vectorized approach is hard for TBL, using iteration for clarity and correctness first.
    # Can optimize later if slow.
    
    # Pre-calculate barriers for all t
    # But barrier depends on Entry Price at t.
    
    # We only care about counts for the distribution, so we can subsample if needed.
    # But let's try full pass.
    
    for t in range(n - horizon):
        entry_price = close_prices[t]
        tp_price = entry_price * (1 + barrier)
        sl_price = entry_price * (1 - barrier)
        
        # Look forward 'horizon' days
        future_highs = high_prices[t+1 : t+1+horizon]
        future_lows = low_prices[t+1 : t+1+horizon]
        
        # Check for touches
        tp_hit_indices = np.where(future_highs >= tp_price)[0]
        sl_hit_indices = np.where(future_lows <= sl_price)[0]
        
        first_tp = tp_hit_indices[0] if len(tp_hit_indices) > 0 else 99999
        first_sl = sl_hit_indices[0] if len(sl_hit_indices) > 0 else 99999
        
        if first_tp == 99999 and first_sl == 99999:
            labels[t] = 1 # No touch
        elif first_tp < first_sl:
            labels[t] = 2 # TP hit first
        elif first_sl < first_tp:
            labels[t] = 0 # SL hit first
        elif first_tp == first_sl:
            # Both hit on same day (first touch day is same)
            labels[t] = 1 # Paper says "If both... hit... on the same day -> Time Limit"
            
    # Remove last 'horizon' labels as they are invalid
    valid_labels = labels[:-horizon]
    
    counts = np.bincount(valid_labels, minlength=3)
    return counts

def evaluate_params(params):
    """Evaluate a single (Horizon, Barrier) pair on a sample of stocks."""
    horizon, barrier, sample_tickers = params
    
    total_counts = np.zeros(3, dtype=int)
    
    # Using a fresh connection per process
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    
    for ticker in sample_tickers:
        try:
            df = pd.read_sql_query(f"SELECT close, high, low FROM prices WHERE ticker = '{ticker}' ORDER BY date", conn)
            if len(df) < horizon + 10:
                continue
            
            counts = apply_triple_barrier(df, horizon, barrier)
            total_counts += counts
        except Exception:
            continue
            
    conn.close()
    return (horizon, barrier, total_counts)

def main():
    timed_logger = TimedLogger()
    timed_logger.log("Starting Triple Barrier Parameter Optimization")
    logger.info("Starting Triple Barrier Parameter Optimization for IHSG")

    # 1. Get tickers
    timed_logger.log("Loading tickers from database")
    tickers = get_tickers()
    logger.info(f"Found {len(tickers)} tickers in database.")
    
    # 2. Sample tickers to speed up (e.g., top 100 liquid ones if we knew them, or random 100)
    # Since we don't have liquidity info easily here without complex query, let's just take a random sample of 50.
    import random
    random.seed(42)
    sample_tickers = random.sample(tickers, min(len(tickers), 50))
    timed_logger.log(f"Using {len(sample_tickers)} tickers for optimization")
    logger.info(f"Using sample of {len(sample_tickers)} tickers for optimization.")

    # 3. Define Parameter Grid
    # Paper: Horizon 5-29, Barrier 7%-15%
    # We will expand slightly: Horizon 5-60, Barrier 3%-20%
    horizons = [5, 10, 15, 20, 25, 29, 30, 40, 50, 60]
    barriers = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.20]
    
    param_grid = []
    for h in horizons:
        for b in barriers:
            param_grid.append((h, b, sample_tickers))

    timed_logger.log(f"Testing {len(param_grid)} configurations")
    logger.info(f"Testing {len(param_grid)} parameter combinations...")

    # 4. Run Optimization
    results = []
    # Use ProcessPoolExecutor for CPU bound tasks
    # Adjust max_workers as needed
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in tqdm(executor.map(evaluate_params, param_grid), total=len(param_grid)):
            results.append(res)
            
    # 5. Analyze Results
    best_score = float('inf')
    best_params = None
    best_dist = None
    
    print("\n--- Top 10 Configurations ---")
    print(f"{'Horizon':<8} {'Barrier':<8} {'Label 0 (SL)':<15} {'Label 1 (Hold)':<15} {'Label 2 (TP)':<15} {'Imbalance Score'}")
    
    # Convert to DataFrame for easier sorting
    data = []
    for h, b, counts in results:
        total = np.sum(counts)
        if total == 0: continue
        
        props = counts / total
        # Metric: Minimize standard deviation from uniform distribution (0.33, 0.33, 0.33)
        # Or minimize Max - Min proportion
        
        # Paper says "balanced label proportions".
        # Ideal is 1/3, 1/3, 1/3.
        # Score = Sum((prop - 1/3)^2)
        score = np.sum((props - 1/3)**2)
        
        data.append({
            'Horizon': h,
            'Barrier': b,
            'Counts': counts,
            'Props': props,
            'Score': score
        })
        
    df_res = pd.DataFrame(data).sort_values('Score')
    
    for idx, row in df_res.head(10).iterrows():
        p = row['Props']
        print(f"{row['Horizon']:<8} {row['Barrier']:<8.2%} {p[0]:<15.1%} {p[1]:<15.1%} {p[2]:<15.1%} {row['Score']:.4f}")
        
    best = df_res.iloc[0]
    timed_logger.log(f"Best config found: Horizon={best['Horizon']}, Barrier={best['Barrier']:.1%}")
    print("\n--- BEST CONFIGURATION ---")
    print(f"Horizon: {best['Horizon']} days")
    print(f"Barrier: {best['Barrier']:.1%}")
    print(f"Distribution: SL={best['Props'][0]:.1%}, Hold={best['Props'][1]:.1%}, TP={best['Props'][2]:.1%}")
    
    # Create a small output file to read from easily
    with open("tbl_params.txt", "w") as f:
        f.write(f"HORIZON={best['Horizon']}\n")
        f.write(f"BARRIER={best['Barrier']}\n")

if __name__ == "__main__":
    main()
