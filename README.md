# Polymarket NLP Lab

Tests whether external sentiment helps predict short-horizon Polymarket price moves using Teichmann's randomized-signature features and ridge regression.

**What it does:** Scans markets by keyword and date overlap, aligns sentiment with price history, builds path features via a CDE, fits ridge models for reconstruction and return prediction and backtests with fees and walk-forward validation.

**How to run** (from repo root; create `out/` and optionally `data/api/` as needed):

```bash
# 1) Scan markets
PYTHONPATH=scripts python3 scripts/scan_overlap_markets.py \
  --start 2026-02-08T00:00:00Z --end 2026-02-22T00:00:00Z \
  --keywords "ukraine,russia,putin,zelensky,ceasefire,nato" \
  --max_offsets 8 --limit 500 --include_closed \
  --out_csv out/overlap_scan.csv

# 2) Run model (one market)
PYTHONPATH=scripts python3 scripts/teichmann_rigorous_pipeline.py \
  --market_id 540816 \
  --sentiment_csv out/gdelt_ukraine_intensity_1h_feb8_feb22.csv \
  --sentiment_ts_col ts --sentiment_value_col sentiment \
  --dt_s 3600 --window 48 --lag 3 --horizon 3 --step 1 \
  --sig_order 3 --n_list 8,12 --depth_two --activation exp \
  --out_prefix teichmann_ukr_feb

# 3) Backtest (walk-forward + fee stress)
PYTHONPATH=scripts python3 scripts/teichmann_backtest.py \
  --market_ids 540816,665224,567689,693519,681144 \
  --sentiment_csv out/gdelt_ukraine_intensity_1h_feb8_feb22.csv \
  --sentiment_ts_col ts --sentiment_value_col sentiment \
  --dt_s 3600 --window 48 --lag 3 --horizon 3 --step 3 \
  --n_list 8,12 --depth_two --activation exp --walk_forward \
  --fee_bps_list 5,10,20 --out_prefix teichmann_bt_top5
```

**Layout:** `scripts/` — Python pipeline; `docs/` — method notes. Outputs go to `out/` and `data/api/`.
