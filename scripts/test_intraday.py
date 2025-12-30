#!/usr/bin/env python3
"""
Test script to check if Yahoo Finance provides intraday data for IDX stocks.
Tests hourly (1h) data availability for a sample of liquid stocks.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Test with liquid IDX stocks
test_tickers = ['BBCA.JK', 'BBRI.JK', 'TLKM.JK', 'ASII.JK', 'UNVR.JK']

print("=" * 80)
print("TESTING INTRADAY DATA AVAILABILITY FOR IDX STOCKS")
print("=" * 80)

for ticker in test_tickers:
    print(f"\n📊 Testing {ticker}...")

    try:
        stock = yf.Ticker(ticker)

        # Test 1: Try to get 1 day of 1-hour data
        print(f"  Attempting 1h interval (hourly)...")
        df_1h = stock.history(period='1d', interval='1h')

        if not df_1h.empty:
            print(f"  ✅ SUCCESS! Got {len(df_1h)} hourly bars")
            print(f"     Date range: {df_1h.index[0]} to {df_1h.index[-1]}")
            print(f"     Sample data:\n{df_1h.head(3)[['Open', 'High', 'Low', 'Close', 'Volume']]}")
        else:
            print(f"  ❌ FAILED - No 1h data returned")

        # Test 2: Try 5-day period with 1h interval
        print(f"\n  Attempting 5d period with 1h interval...")
        df_5d = stock.history(period='5d', interval='1h')

        if not df_5d.empty:
            print(f"  ✅ SUCCESS! Got {len(df_5d)} hourly bars across 5 days")
            print(f"     Unique dates: {df_5d.index.date}")
        else:
            print(f"  ❌ FAILED - No 5d/1h data returned")

        # Test 3: Try 60-day period to check historical availability
        print(f"\n  Attempting 60d period with 1h interval...")
        df_60d = stock.history(period='60d', interval='1h')

        if not df_60d.empty:
            print(f"  ✅ SUCCESS! Got {len(df_60d)} hourly bars")
            print(f"     Date range: {df_60d.index[0]} to {df_60d.index[-1]}")
        else:
            print(f"  ❌ FAILED - No 60d/1h data returned")

    except Exception as e:
        print(f"  ❌ ERROR: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
