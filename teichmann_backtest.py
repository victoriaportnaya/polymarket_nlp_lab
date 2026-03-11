import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from teichmann_rigorous_pipeline import (
    RandomizedSignatureExtractor,
    _parse_gamma_list_field,
    align_sentiment_to_index,
    fetch_clob_prices_history,
    fetch_gamma_market_by_id,
    load_sentiment_series,
    regularize_prices,
    ridge_fit_predict,
)


def _ensure_dirs() -> None:
    os.makedirs("out", exist_ok=True)
    os.makedirs("data/api", exist_ok=True)
    mpl_cfg = os.path.abspath(".mplconfig")
    os.makedirs(mpl_cfg, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)
    os.environ.setdefault("MPLBACKEND", "Agg")


def _pick_yes_token(market: dict) -> tuple[str, str]:
    outcomes = _parse_gamma_list_field(market.get("outcomes"))
    tids = _parse_gamma_list_field(market.get("clobTokenIds"))
    if not (len(outcomes) == 2 and len(tids) == 2):
        raise ValueError("Market is not a simple 2-outcome CLOB market.")
    idx = 0
    if outcomes[0].lower() == "no" and outcomes[1].lower() == "yes":
        idx = 1
    elif outcomes[0].lower() == "yes":
        idx = 0
    elif outcomes[1].lower() == "yes":
        idx = 1
    return str(tids[idx]), str(outcomes[idx])


@dataclass
class WindowPack:
    X_windows: list[np.ndarray]
    y_fwdret: np.ndarray
    ts_entry: list[pd.Timestamp]
    ts_exit: list[pd.Timestamp]


def _clip01(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def build_windows_for_bt(
    s: np.ndarray,
    p: np.ndarray,
    ts: pd.Series,
    window: int,
    lag: int,
    horizon: int,
    step: int,
) -> WindowPack:
    p = _clip01(np.asarray(p, float))
    s = np.asarray(s, float)
    n = len(p)
    Xs: list[np.ndarray] = []
    rets: list[float] = []
    t_entry: list[pd.Timestamp] = []
    t_exit: list[pd.Timestamp] = []
    start = window + lag
    end = n - horizon
    for t in range(start, end, step):
        sl = slice(t - window, t + 1)
        s_path = s[sl]
        p_path = p[sl.start - lag : sl.stop - lag]
        if len(p_path) != len(s_path):
            continue
        Xs.append(np.column_stack([s_path, p_path]))
        rets.append(float(np.log(p[t + horizon]) - np.log(p[t])))
        t_entry.append(pd.Timestamp(ts.iloc[t]))
        t_exit.append(pd.Timestamp(ts.iloc[t + horizon]))
    return WindowPack(X_windows=Xs, y_fwdret=np.asarray(rets, float), ts_entry=t_entry, ts_exit=t_exit)


def run_bt(y_true: np.ndarray, y_pred: np.ndarray, fee_bps: float) -> dict:
    pos = np.sign(y_pred).astype(float)
    fee = fee_bps / 10000.0
    costs = np.zeros_like(pos)
    if len(pos) > 0:
        costs[0] = fee * abs(pos[0])
        costs[1:] = fee * np.abs(pos[1:] - pos[:-1])
    strat = pos * y_true - costs
    eq = np.exp(np.cumsum(strat))
    ann = float(np.sqrt(24.0 * 365.0))
    mu = float(np.mean(strat)) if len(strat) else float("nan")
    sd = float(np.std(strat, ddof=1)) if len(strat) > 1 else float("nan")
    sharpe = float(mu / (sd + 1e-12) * ann) if np.isfinite(mu) and np.isfinite(sd) else float("nan")
    hit = float(np.mean((np.sign(y_true) == np.sign(y_pred)).astype(float))) if len(y_true) else float("nan")
    max_dd = float(np.max(1.0 - eq / np.maximum.accumulate(eq))) if len(eq) else float("nan")
    return {
        "n_test": int(len(y_true)),
        "cum_return": float(eq[-1] - 1.0) if len(eq) else float("nan"),
        "mean_logret": mu,
        "vol_logret": sd,
        "sharpe_ann": sharpe,
        "hit_rate": hit,
        "max_drawdown": max_dd,
        "avg_turnover": float(np.mean(np.abs(np.diff(pos)))) if len(pos) > 1 else 0.0,
    }, strat, eq, pos


def run_walk_forward_predictions(Phi: np.ndarray, y: np.ndarray, train_frac: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    n_total = len(y)
    ntr = int(train_frac * n_total)
    if ntr < 10 or n_total - ntr < 10:
        raise ValueError(f"Not enough data for walk-forward split: n_total={n_total}, n_train={ntr}")
    preds = []
    yt = []
    for t in range(ntr, n_total):
        yhat, _ = ridge_fit_predict(Phi[:t], y[:t].reshape(-1, 1), Phi[t : t + 1], lam=lam)
        preds.append(float(yhat.reshape(-1)[0]))
        yt.append(float(y[t]))
    return np.asarray(yt, float), np.asarray(preds, float)


def main() -> None:
    _ensure_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--market_ids", type=str, required=True, help="Comma-separated Gamma market IDs")
    ap.add_argument("--sentiment_csv", type=str, required=True)
    ap.add_argument("--sentiment_ts_col", type=str, default="ts")
    ap.add_argument("--sentiment_value_col", type=str, default="sentiment")
    ap.add_argument("--sentiment_align", type=str, default="ffill")
    ap.add_argument("--dt_s", type=int, default=3600)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--lag", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--vf_scale", type=float, default=0.005)
    ap.add_argument("--m_init", type=int, default=16)
    ap.add_argument("--n_list", type=str, default="8,12")
    ap.add_argument("--depth_two", action="store_true")
    ap.add_argument("--activation", type=str, default="exp", choices=["exp", "tanh"])
    ap.add_argument("--n_draws", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fee_bps", type=float, default=5.0, help="Used if --fee_bps_list is empty")
    ap.add_argument("--fee_bps_list", type=str, default="5,10,20", help="Comma-separated fees in bps")
    ap.add_argument("--walk_forward", action="store_true", help="Use walk-forward retraining on test period")
    ap.add_argument("--out_prefix", type=str, default="teichmann_bt")
    args = ap.parse_args()

    s_raw = load_sentiment_series(args.sentiment_csv, ts_col=args.sentiment_ts_col, value_col=args.sentiment_value_col)
    sentiment_start = s_raw.index.min()
    sentiment_end = s_raw.index.max()
    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    mids = [x.strip() for x in args.market_ids.split(",") if x.strip()]
    fee_list = [float(x) for x in args.fee_bps_list.split(",") if x.strip()] or [float(args.fee_bps)]
    rows = []

    for mid in mids:
        market_path = f"data/api/gamma_market_{mid}.json"
        if os.path.exists(market_path):
            with open(market_path, "r", encoding="utf-8") as f:
                market = json.load(f)
        else:
            market = fetch_gamma_market_by_id(mid)
            with open(market_path, "w", encoding="utf-8") as f:
                json.dump(market, f, ensure_ascii=False, indent=2)

        token_id, outcome_name = _pick_yes_token(market)
        buf = pd.Timedelta(days=1)
        dfp = fetch_clob_prices_history(
            token_id,
            interval="max",
            start_ts=sentiment_start - buf,
            end_ts=sentiment_end + buf,
        )
        dfp = regularize_prices(dfp, dt_s=args.dt_s)
        dfp = dfp[(dfp["ts"] >= sentiment_start - buf) & (dfp["ts"] <= sentiment_end + buf)].copy()
        s_al = align_sentiment_to_index(s_raw, pd.DatetimeIndex(dfp["ts"]), method=args.sentiment_align)
        s = s_al.astype(float).to_numpy()
        s = (s - float(np.mean(s))) / (float(np.std(s)) + 1e-12)
        p = dfp["price"].astype(float).to_numpy()
        wp = build_windows_for_bt(
            s=s,
            p=p,
            ts=dfp["ts"],
            window=args.window,
            lag=args.lag,
            horizon=args.horizon,
            step=args.step,
        )
        n_total = len(wp.X_windows)
        ntr = int(args.train_frac * n_total)
        Xtr, Xte = wp.X_windows[:ntr], wp.X_windows[ntr:]
        ytr, yte = wp.y_fwdret[:ntr], wp.y_fwdret[ntr:]
        ts_entry_te = wp.ts_entry[ntr:]
        ts_exit_te = wp.ts_exit[ntr:]

        for depth_tag in ([False, True] if args.depth_two else [False]):
            for n in n_list:
                for draw in range(args.n_draws):
                    seed = args.seed + 1000 * draw + (17 if depth_tag else 0) + 3 * n
                    extractor = RandomizedSignatureExtractor(
                        d=2,
                        n=n,
                        m_init=args.m_init,
                        seed=seed,
                        depth_two=depth_tag,
                        vf_scale=args.vf_scale,
                        activation=args.activation,
                    )
                    Phi_tr = np.vstack([extractor.features(X) for X in Xtr])
                    Phi_te = np.vstack([extractor.features(X) for X in Xte])
                    if args.walk_forward:
                        Phi_all = np.vstack([extractor.features(X) for X in wp.X_windows])
                        y_eval, yhat = run_walk_forward_predictions(Phi_all, wp.y_fwdret, train_frac=args.train_frac, lam=args.lam)
                        ts_entry_eval = wp.ts_entry[ntr:]
                        ts_exit_eval = wp.ts_exit[ntr:]
                        model_tag = "walkfwd"
                    else:
                        yhat, _ = ridge_fit_predict(Phi_tr, ytr.reshape(-1, 1), Phi_te, lam=args.lam)
                        yhat = yhat.reshape(-1)
                        y_eval = yte
                        ts_entry_eval = ts_entry_te
                        ts_exit_eval = ts_exit_te
                        model_tag = "static"

                    for fee in fee_list:
                        bt, strat, eq, pos = run_bt(y_eval, yhat, fee_bps=fee)
                        rows.append(
                            {
                                "market_id": mid,
                                "question": market.get("question", ""),
                                "token_id": token_id,
                                "outcome": outcome_name,
                                "depth_two": bool(depth_tag),
                                "N": n,
                                "draw": draw,
                                "model_type": model_tag,
                                "fee_bps": fee,
                                **bt,
                            }
                        )
                        detail = pd.DataFrame(
                            {
                                "ts_entry": ts_entry_eval,
                                "ts_exit": ts_exit_eval,
                                "y_true": y_eval,
                                "y_pred": yhat,
                                "position": pos,
                                "strategy_logret": strat,
                                "equity": eq,
                            }
                        )
                        detail.to_csv(
                            f"out/{args.out_prefix}_mkt{mid}_N{n}_d{int(depth_tag)}_draw{draw}_{model_tag}_fee{int(fee)}_trades.csv",
                            index=False,
                        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(f"out/{args.out_prefix}_summary.csv", index=False)
    agg = (
        out_df.groupby(["market_id", "fee_bps"], as_index=False)
        .agg(
            best_cum_return=("cum_return", "max"),
            best_sharpe=("sharpe_ann", "max"),
            median_cum_return=("cum_return", "median"),
            median_sharpe=("sharpe_ann", "median"),
        )
        .sort_values(["fee_bps", "best_sharpe"], ascending=[True, False])
    )
    agg.to_csv(f"out/{args.out_prefix}_aggregate.csv", index=False)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Plot best equity curve per market by Sharpe.
    for mid in sorted(out_df["market_id"].unique().tolist()):
        d = out_df[out_df["market_id"] == mid].copy()
        if d.empty:
            continue
        best = d.sort_values("sharpe_ann", ascending=False).iloc[0]
        p = (
            f"out/{args.out_prefix}_mkt{mid}_N{int(best['N'])}_d{int(best['depth_two'])}_draw{int(best['draw'])}_"
            f"{best['model_type']}_fee{int(best['fee_bps'])}_trades.csv"
        )
        tr = pd.read_csv(p)
        plt.figure(figsize=(8, 3.2))
        plt.plot(pd.to_datetime(tr["ts_exit"], utc=True), tr["equity"])
        plt.title(
            f"Best equity market {mid} ({best['model_type']}, fee {int(best['fee_bps'])}bps, Sharpe {best['sharpe_ann']:.2f})"
        )
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(f"out/{args.out_prefix}_mkt{mid}_best_equity.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()

