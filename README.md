# Polymarket Sentiment Signals

This repository tests whether external sentiment can help predict short-horizon Polymarket price moves.

## What it does

It provides an end-to-end research pipeline:

1. Find topic-relevant Polymarket markets by keyword and date overlap.
2. Align sentiment time series with market price history.
3. Build path features using Teichmann-style randomized signatures (CDE-based).
4. Train ridge models for:
   - signature reconstruction,
   - forward-return prediction.
5. Backtest signal quality out-of-sample with transaction costs and walk-forward retraining.

## Main scripts

- `scan_overlap_markets.py`  
  Scans Gamma/CLOB for markets matching your keywords and time window.

- `teichmann_rigorous_pipeline.py`  
  Core modeling script for reconstruction and prediction metrics.

- `teichmann_backtest.py`  
  Backtesting script with fee stress (`5/10/20 bps`) and walk-forward option.

- `sentiment_io.py`  
  Sentiment loading and timestamp alignment utilities.

## Quick start

### 1) Scan markets by topic and overlap

```bash
python3 scan_overlap_markets.py \
  --start 2026-02-08T00:00:00Z \
  --end 2026-02-22T00:00:00Z \
  --keywords "ukraine,russia,putin,zelensky,ceasefire,nato" \
  --max_offsets 8 \
  --limit 500 \
  --include_closed \
  --out_csv out/overlap_scan_ukraine_feb8_feb22_2026.csv
```

### 2) Run core modeling on one market

```bash
python3 teichmann_rigorous_pipeline.py \
  --market_id 540816 \
  --sentiment_csv out/gdelt_ukraine_intensity_1h_feb8_feb22.csv \
  --sentiment_ts_col ts \
  --sentiment_value_col sentiment \
  --dt_s 3600 \
  --min_points 200 \
  --window 48 --lag 3 --horizon 3 --step 1 \
  --sig_order 3 \
  --n_list 8,12 \
  --depth_two \
  --activation exp \
  --n_draws 2 \
  --out_prefix teichmann_ukr_feb
```

### 3) Run robust backtest on top markets

```bash
python3 teichmann_backtest.py \
  --market_ids 540816,665224,567689,693519,681144 \
  --sentiment_csv out/gdelt_ukraine_intensity_1h_feb8_feb22.csv \
  --sentiment_ts_col ts \
  --sentiment_value_col sentiment \
  --dt_s 3600 \
  --window 48 --lag 3 --horizon 3 --step 3 \
  --n_list 8,12 \
  --depth_two \
  --activation exp \
  --n_draws 2 \
  --walk_forward \
  --fee_bps_list 5,10,20 \
  --out_prefix teichmann_ukr_feb_bt_top5
```

## Outputs

- `out/*overlap_scan*.csv`: market candidates
- `data/api/clob_prices_*`: fetched price history
- `out/*reconstruction_metrics.csv`: reconstruction quality
- `out/*prediction_metrics.csv`: prediction metrics
- `out/*_summary.csv`, `out/*_aggregate.csv`: backtest summaries
- `out/*.png`: plots (IC, R2, equity curves)

# Polymarket + Sentiment Research (Human Guide)

This repo is a practical pipeline for one question:

**Can external sentiment streams help predict Polymarket price moves?**

It does this with Teichmann-style randomized-signature features (CDE-based) and then tests if the signal survives out-of-sample and with trading costs.

---

## What is in this project

- Pull market candidates by keyword + time overlap
- Pull Polymarket price history for selected markets
- Build sentiment time series (Exorde / GDELT / other inputs)
- Run Teichmann-style randomized-signature modeling
- Evaluate:
  - signature reconstruction quality
  - predictive quality (IC / R2)
  - trading backtest (PnL, Sharpe, drawdown, fee stress)

---

## End-to-end logic (simple version)

1. **Start with a sentiment time series** (hourly/daily).
2. **Find markets that match your topic keywords** and overlap that time range.
3. **Fetch market price history** and align it to sentiment timestamps.
4. Build path windows `X = [sentiment, lagged_price]`.
5. Extract randomized-signature features from a CDE with random vector fields.
6. Fit linear ridge models:
   - reconstruction target (signature components),
   - prediction target (future return).
7. Run backtest:
   - position = sign(predicted return),
   - include fees/slippage assumptions,
   - report robust metrics.

---

## Main scripts and what each does

### Core modeling and evaluation

- `teichmann_rigorous_pipeline.py`  
  Main experiment runner for reconstruction + prediction metrics.
  - Supports Teichmann-style depth-two exponential fields (`--activation exp`).
  - Supports fallback for tokens where bounded `prices-history` queries fail.
  - Supports `--min_points` to control minimum required overlap.

- `teichmann_backtest.py`  
  Out-of-sample trading backtest on top of model predictions.
  - Static split or walk-forward (`--walk_forward`)
  - Fee sweep (`--fee_bps_list`, e.g. `5,10,20`)
  - Saves per-trade logs and aggregate summaries.

### Market selection and data alignment

- `scan_overlap_markets.py`  
  Keyword scan over Gamma markets + overlap check with CLOB price history.
  - Uses safer keyword matching (word boundaries) to reduce false positives.

- `sentiment_io.py`  
  Utilities to load sentiment series and align to target timestamps.

### Sentiment data prep

- `exorde_sentiment_eda.py`  
  Build/clean sentiment from Exorde exports, resample, and visualize.

- `gdelt_sentiment.py`  
  Build GDELT-based sentiment/intensity series.

- `wikipedia_pageviews_sentiment.py`  
  Optional alternate sentiment proxy from Wikipedia activity.

### Supporting scripts

- `run_exorde_to_teichmann.py`  
  Convenience launcher from Exorde preprocessing to model runs.

- `polymarket_signature_experiments.py`  
  Signature-based lead/lag style experiments (legacy/auxiliary).

- `polymarket_randomized_signature_reconstruction.py`  
  Earlier randomized-signature reconstruction script (legacy/auxiliary).

- `polymarket_sde_experiment.py`  
  Experimental SDE-related path modeling (auxiliary).

- `clob_trades.py`, `clob_l2_auth.py`  
  Trade-level CLOB access (authenticated path when needed).

---

## Typical workflow (recommended)

### 1) Create / pick a sentiment series

Use an existing file in `out/` (for example):

- `out/gdelt_ukraine_intensity_1h_feb8_feb22.csv`
- `out/exorde_ukrainewar_sentiment_1h.csv`

Expected columns:
- `ts`
- `sentiment`

### 2) Find relevant markets in same date range

```bash
python3 scan_overlap_markets.py \
  --start 2026-02-08T00:00:00Z \
  --end 2026-02-22T00:00:00Z \
  --keywords "ukraine,russia,putin,zelensky,ceasefire,nato" \
  --max_offsets 8 \
  --limit 500 \
  --include_closed \
  --out_csv out/overlap_scan_ukraine_feb8_feb22_2026.csv
```

### 3) Run Teichmann modeling on selected markets

```bash
python3 teichmann_rigorous_pipeline.py \
  --market_id 540816 \
  --sentiment_csv out/gdelt_ukraine_intensity_1h_feb8_feb22.csv \
  --sentiment_ts_col ts \
  --sentiment_value_col sentiment \
  --dt_s 3600 \
  --min_points 200 \
  --window 48 --lag 3 --horizon 3 --step 1 \
  --sig_order 3 \
  --n_list 8,12 \
  --depth_two \
  --activation exp \
  --n_draws 2 \
  --out_prefix teichmann_ukr_feb
```

### 4) Run robust backtest (walk-forward + fee stress)

```bash
python3 teichmann_backtest.py \
  --market_ids 540816,665224,567689,693519,681144 \
  --sentiment_csv out/gdelt_ukraine_intensity_1h_feb8_feb22.csv \
  --sentiment_ts_col ts \
  --sentiment_value_col sentiment \
  --dt_s 3600 \
  --window 48 --lag 3 --horizon 3 --step 3 \
  --n_list 8,12 \
  --depth_two \
  --activation exp \
  --n_draws 2 \
  --walk_forward \
  --fee_bps_list 5,10,20 \
  --out_prefix teichmann_ukr_feb_bt_top5
```

---

## How to read results

### Modeling metrics

- **Reconstruction mean R2**  
  How well randomized-signature features recover signature targets.
  - Higher is better.
  - Good reconstruction does **not** automatically mean good prediction.

- **IC (information coefficient)**  
  Correlation between predicted and realized forward return.
  - Near 0: weak signal
  - Positive: directional edge

- **Predictive R2**
  - Positive: model explains some return variance
  - Negative: worse than a naive mean prediction

### Backtest metrics

- **cum_return**: compounded return over test period
- **sharpe_ann**: annualized Sharpe (use carefully on short windows)
- **hit_rate**: directional accuracy
- **max_drawdown**: worst peak-to-trough loss
- **avg_turnover**: how often position changes (cost sensitivity)

---

## Where outputs go

- `out/*overlap_scan*.csv` -> market candidates by keyword/time overlap
- `data/api/clob_prices_*` -> fetched regularized price histories
- `out/*reconstruction_metrics.csv` -> reconstruction stats
- `out/*prediction_metrics.csv` -> IC/R2 stats
- `out/*_summary.csv` -> backtest run-level results
- `out/*_aggregate.csv` -> aggregated robust results by market/fee
- `out/*.png` -> plots (reconstruction, IC, R2, equity curves)

---

## Notes and caveats

- Short windows can produce inflated Sharpe.
- Prefer non-overlapping windows for cleaner backtests (`step >= horizon`).
- Always stress with higher fees (`10-20 bps`) and multiple markets.
- One strong market is not enough; look for consistency across market basket.

---

## Minimal rule of thumb

If a setup shows:
- positive IC,
- mostly positive net returns after fees,
- survives fee increase and multiple markets,

then keep it for further testing. Otherwise treat it as exploratory noise.

