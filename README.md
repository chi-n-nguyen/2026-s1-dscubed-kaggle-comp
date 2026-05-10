# 2026 S1 DSCubed Kaggle Stock Market Prediction Competition 2nd Place Solution

9 May 2026 

## Problem

Given 40 historical price steps (`p1`–`p40`) for each of 10,000 stocks, predict which stocks will decline by the next observation (`p50`) and decide which to sell. The objective is to maximise total profit from correct sell decisions, subject to a sell quota constraint derived from the submission template.

## Approach

### Feature Engineering (3 tiers)

Features are built exclusively from the 40-step price series. Each tier adds complexity and is only included if it improves out-of-fold profit.

- **v1 - Price levels & returns:** last/first/mean price, total and windowed returns (5, 10 step), return statistics, simple moving averages (5, 10, 20), MA crossovers, rolling volatility, up/down ratio.
- **v2 - Trend & momentum:** linear slope via `polyfit` over multiple windows, momentum acceleration (short vs long window return difference), position within recent high-low range, max drawdown from running peak, extreme move detection.
- **v3 - Technical indicators:** log return distribution (mean, std, skew, kurtosis), RSI at 5/10/14 periods, Bollinger band position, exponential moving averages with price-EMA spread, return autocorrelation at lags 1/3/5, price ratios to lagged values.

### Models

- **XGBoost** (`reg:squarederror`, MAE eval, depth 6, lr 0.05, L1+L2 reg)
- **LightGBM** (same general config, 31 leaves)
- **Ensemble** — simple average of XGB and LGB predictions. Only submitted if OOF profit exceeds both individual models.

All models trained with 5-fold CV. Test predictions averaged across folds.

### Sample Weighting

Inspired by [Dr Yirun Zhang's 1st place solution](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s) in the 2021 Jane Street Market Prediction competition. Training samples are weighted by absolute price change magnitude, clipped at the 95th percentile and normalised to mean 1. This forces the model to prioritise stocks with large moves that contribute most to profit.

### Sell Count Optimisation

Rather than using the default sell quota, the pipeline sweeps sell ratios from 10% to 50% on out-of-fold predictions and selects the count that maximises OOF profit. A higher hold ratio produced lower raw profit on paper but higher confidence per trade, and selling fewer stocks with stronger conviction outperformed aggressive selling on the leaderboard.

## Pipeline Structure

```
Load data -> Holdout split (7%) -> Build features (v1/v2/v3)
  -> Train XGB -> Train XGB weighted -> Train LGB -> Ensemble
  -> Compare all OOF profits -> Sell count sweep on best model
  -> Generate final submission
```

Each model is only submitted if it beats the current best OOF profit. Baselines (momentum heuristic, zero predictor) are evaluated first as sanity checks.

## Validation

- **5-fold CV** with OOF profit as the primary selection metric
- **7% holdout** reserved as a final sanity check, never used for model selection
- **Sign validation** to confirm sell decisions target negative predicted price changes

## Results

[Full leaderboard](https://www.kaggle.com/competitions/ds-cubed-kaggle-competition-2026-s-1/leaderboard) · 2nd place out of 30+ teams with a score of 18,944.25 (profit), within 512.75 of 1st place. Solo entry with 6 submissions.

## Requirements

```
numpy
pandas
xgboost
lightgbm
scikit-learn
```

## Usage

```bash
# Place train.csv, test.csv, sample_submission.csv in the working directory
jupyter notebook competition_notebook_final.ipynb
# Run all cells — submissions are saved automatically
```
