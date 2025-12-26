# How to Build an AI Trading System: The Paperium Blueprint

When someone asks "How do you build a system that trades stocks automatically?", the answer isn't just "Write some code." It is about building a machine that can see, think, and protect itself.

Here is the step-by-step process of how we built Paperium for the Indonesia Stock Exchange (IHSG).

---

## Step 1: Gathering the Raw Materials (Data)

The first thing you need is high-quality history. We built a pipeline that connects to market data providers (like Yahoo Finance) to download Open, High, Low, Close, and Volume (OHLCV) data for hundreds of stocks. 

Without clean data, the machine has nothing to learn from. We store this in a structured database so we can access years of history in seconds.

---

## Step 2: Giving the Machine 'Eyes' (Feature Engineering)

A computer doesn't "see" a price chart like we do. To help it understand what's happening, we have to translate prices into mathematical patterns. We call these "Features."

If you were building this, you would create 46 different "eyes" for your machine. Some see trends, some see volatility, and some see momentum.

### The Math Behind the Sight

To make the data understandable for the machine, we use specific formulas:

**1. Log Returns: Measuring Velocity**
We don't just look at the price change; we look at the "Log Return." This helps the machine compare a stock that costs 100 with one that costs 10,000 on the same scale.
$$r_t = \ln\left(\frac{Price_t}{Price_{t-1}}\right)$$

**2. Volatility Z-Score: Identifying Chaos**
We want the machine to know if the market is behaving "strangely." The Z-Score tells us how many standard deviations the current volatility is away from its normal average.
$$Z_{\sigma} = \frac{\sigma - \mu_{\sigma}}{\text{std}(\sigma)}$$

**3. Relative Volume: Detecting Crowds**
Is everyone trading this stock today, or is it quiet? We compare today's volume to the 20-day average.
$$RelVol = \frac{Volume_t}{SMA(Volume, 20)}$$

---

## Step 3: Building the Brain (XGBoost)

Now that the machine can "see" patterns, we need it to make decisions. For Paperium, we chose **XGBoost (Extreme Gradient Boosting)**.

### How does this brain work?
Think of XGBoost as a "Committee of Experts." 
- We create hundreds of small "Decision Trees." 
- Each tree looks at the 46 features and tries to guess: "Will this stock go up tomorrow?"
- If the first tree makes a mistake, the second tree focuses specifically on fixing that mistake. 
- By the time we reach the 100th tree, the committee is very good at predicting the probability of success.

---

## Step 4: Teaching through History (The Training Loop)

To build this, you can't just train the brain once and leave it. Markets change. 

We use a **Rolling Window** approach. Every day, the machine looks at exactly the last 252 trading days (one year). It "practices" on that data until it finds the best strategy for the *current* market condition.

---

## Step 5: The Survival Kit (Risk Management)

A trading system that doesn't manage risk is just a gambling machine. When building Paperium, we added three layers of protection:

1. **Position Sizing**: The machine calculates exactly how much money to put into each trade so that one bad trade doesn't break the bank.
2. **Trailing Stop Loss**: If a trade starts winning, the "Stop Loss" moves up behind the price like a safety net. If the price falls, we lock in our profits.
3. **Time Stop**: If a stock doesn't move after 5 days, the machine gets bored and sells it to find a better opportunity elsewhere.

---

## Step 6: The Daily Routine (Automation)

The final step in building this is making it run while you sleep.
- **Morning**: The machine looks at the market, runs the 46 features through the XGBoost brain, and gives us a list of "Buy" signals.
- **Evening**: The machine looks at how it performed today, adds today's data to its memory, and retrains itself to be smarter for tomorrow.

---

## Summary

Building an AI trading system is about **Automated Discipline**. By combining data pipelines, mathematical features, and a self-improving brain, we created a system that doesn't get emotional, doesn't get tired, and never stops learning from the IHSG market.
