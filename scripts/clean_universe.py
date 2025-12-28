#!/usr/bin/env python3
import sys
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def clean_universe():
    print(f"Cleaning universe of {len(config.data.stock_universe)} stocks...")
    cutoff_date = datetime.now() - timedelta(days=30)
    active_stocks = []
    failed_tickers = []
    
    # 1. Try batching first in chunks of 20 (smaller is safer for timeouts)
    chunk_size = 20
    for i in range(0, len(config.data.stock_universe), chunk_size):
        chunk = config.data.stock_universe[i:i+chunk_size]
        print(f"  Checking chunk {i//chunk_size + 1} ({len(chunk)} stocks)...")
        
        try:
            data = yf.download(chunk, period="1mo", progress=False, group_by='ticker', timeout=20)
            
            for ticker in chunk:
                try:
                    ticker_data = None
                    if isinstance(data, pd.DataFrame):
                        if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.levels[0]:
                            ticker_data = data[ticker].dropna()
                        else:
                            # Single ticker case if chunk size became 1
                            ticker_data = data.dropna()
                    
                    if ticker_data is not None and not ticker_data.empty:
                        latest_date = ticker_data.index[-1]
                        if latest_date >= cutoff_date:
                            active_stocks.append(ticker)
                        else:
                            print(f"    - {ticker} inactive since {latest_date}")
                    else:
                        print(f"    - {ticker} no recent data in batch, adding to individual retry")
                        failed_tickers.append(ticker)
                except Exception as e:
                    failed_tickers.append(ticker)
        except Exception as e:
            print(f"    Batch error ({e}), adding chunk to individual retry")
            failed_tickers.extend(chunk)

    # 2. Individual retries for failed/timed-out tickers
    if failed_tickers:
        print(f"\nRetrying {len(failed_tickers)} tickers individually...")
        for ticker in failed_tickers:
            try:
                time.sleep(0.5) # Avoid rate limiting
                ticker_df = yf.download(ticker, period="1mo", progress=False, timeout=10)
                if ticker_df is not None and not ticker_df.empty:
                    latest_date = ticker_df.index[-1]
                    if latest_date >= cutoff_date:
                        active_stocks.append(ticker)
                        print(f"    + {ticker} active (found via retry)")
                    else:
                        print(f"    - {ticker} inactive since {latest_date}")
                else:
                    print(f"    - {ticker} definitively no data")
            except Exception as e:
                print(f"    - {ticker} final error: {e}")

    # Remove duplicates but maintain some order if possible
    active_stocks = list(dict.fromkeys(active_stocks))
    
    print(f"\nFinal active count: {len(active_stocks)} (Removed {len(config.data.stock_universe) - len(active_stocks)})")
    
    # Update data/universe.py
    with open('data/universe.py', 'r+') as f:
        content = f.read()
        import re
        
        # Pattern to find IDX_UNIVERSE list
        pattern = r"(IDX_UNIVERSE = )\[.*?\]"
        
        # Format the list nicely for the file
        list_str = "[\n    " + ",\n    ".join([f"'{s}'" for s in active_stocks]) + "\n]"
        replacement = r"\1" + list_str
        
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        f.seek(0)
        f.write(new_content)
        f.truncate()
        
    print("✓ data/universe.py updated with robust active universe.")

if __name__ == "__main__":
    clean_universe()
