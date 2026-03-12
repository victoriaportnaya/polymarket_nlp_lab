import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from sentiment_io import align_sentiment_to_index, load_sentiment_series


def _ensure_dirs() -> None:
    os.makedirs("out", exist_ok=True)
    mpl_cfg = os.path.abspath(".mplconfig")
    os.makedirs(mpl_cfg, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)
    os.environ.setdefault("MPLBACKEND", "Agg")


def _clip01(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _json_get(url: str, params: dict | None = None, timeout_s: int = 30) -> dict | list:
    # sandbox often lacks CA roots; fine for research data pulls
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    r = requests.get(url, params=params, timeout=timeout_s, verify=False)
    r.raise_for_status()
    return r.json()


def fetch_gamma_markets(limit: int = 100) -> list[dict]:
    url = "https://gamma-api.polymarket.com/markets"
    params = {"closed": "false", "limit": str(limit)}
    return _json_get(url, params=params)  # type: ignore[return-value]


def fetch_price_history(token_id: str, interval: str = "max") -> pd.DataFrame:
    url = "https://clob.polymarket.com/prices-history"
    params = {"market": token_id, "interval": interval}
    j = _json_get(url, params=params)  # {"history":[{"t":...,"p":...},...]}
    hist = j.get("history", [])
    df = pd.DataFrame(hist)
    if df.empty:
        return df
    df = df.rename(columns={"t": "ts", "p": "price"})
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.sort_values("ts").drop_duplicates("ts")
    return df[["ts", "price"]]


def regularize_history(df: pd.DataFrame, target_dt_s: int = 60) -> tuple[pd.DataFrame, float]:
    if df.empty:
        return df, float("nan")
    s = df.set_index("ts")["price"].astype(float)
    rule = f"{int(target_dt_s)}s"
    s_reg = s.resample(rule).last().ffill()
    out = s_reg.reset_index().rename(columns={"index": "ts", "price": "price"})
    out = out.dropna()
    return out, float(target_dt_s)


def parse_gamma_list_field(x) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def pick_market(markets: list[dict], prefer_yes_no: bool = True) -> dict:
    def vol(m: dict) -> float:
        try:
            return float(m.get("volumeNum") or 0.0)
        except Exception:
            return 0.0

    markets = sorted(markets, key=vol, reverse=True)
    for m in markets:
        outcomes = parse_gamma_list_field(m.get("outcomes"))
        token_ids = parse_gamma_list_field(m.get("clobTokenIds"))
        if len(outcomes) != 2 or len(token_ids) != 2:
            continue
        if prefer_yes_no and not (outcomes[0].lower() in {"yes", "no"} and outcomes[1].lower() in {"yes", "no"}):
            continue
        return m
    raise RuntimeError("No suitable market found.")


def make_proxy_sentiment(prices: np.ndarray, span: int = 120) -> np.ndarray:
    """
    Proxy sentiment for demo: EWMA momentum of returns (bounded via tanh).
    Replace this with real sentiment aligned to timestamps.
    """
    p = _clip01(prices)
    r = np.diff(np.log(p), prepend=np.log(p[0]))
    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(r)
    ema[0] = r[0]
    for i in range(1, len(r)):
        ema[i] = alpha * r[i] + (1 - alpha) * ema[i - 1]
    z = ema / (np.std(ema) + 1e-12)
    return np.tanh(z)


@dataclass
class Signature:
    level1: np.ndarray  # (d,)
    level2: np.ndarray  # (d,d)


def signature_level2(path: np.ndarray) -> Signature:
    """
    Truncated signature up to level 2 for a piecewise-linear path.
    For each increment v, segment signature is (1, v, 1/2 v⊗v).
    Concatenate via Chen: (a1,a2) ⊗ (b1,b2) -> (a1+b1, a2+b2+a1⊗b1)
    """
    x = np.asarray(path, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        d = x.shape[1] if x.ndim == 2 else 0
        return Signature(level1=np.zeros(d), level2=np.zeros((d, d)))
    d = x.shape[1]
    a1 = np.zeros(d, dtype=float)
    a2 = np.zeros((d, d), dtype=float)
    dx = np.diff(x, axis=0)
    for v in dx:
        b1 = v
        b2 = 0.5 * np.outer(v, v)
        a2 = a2 + b2 + np.outer(a1, b1)
        a1 = a1 + b1
    return Signature(level1=a1, level2=a2)


def signature_features(path: np.ndarray) -> tuple[np.ndarray, dict]:
    sig = signature_level2(path)
    d = sig.level1.shape[0]
    feats = np.concatenate([sig.level1.reshape(-1), sig.level2.reshape(-1)])
    names = {}
    for i in range(d):
        names[i] = f"L1_d{i}"
    k = d
    for i in range(d):
        for j in range(d):
            names[k] = f"L2_d{i}{j}"
            k += 1
    return feats, names


def ridge_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, lam: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    """
    Ridge regression with intercept via augmented design matrix.
    Returns (y_pred, coef_with_intercept).
    """
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    ytr = np.asarray(ytr, float).reshape(-1)
    Xtr_aug = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
    Xte_aug = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
    # Do not penalize intercept:
    I = np.eye(Xtr_aug.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.solve(Xtr_aug.T @ Xtr_aug + lam * I, Xtr_aug.T @ ytr)
    yhat = Xte_aug @ beta
    return yhat, beta


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build_dataset(
    ts: np.ndarray,
    price: np.ndarray,
    sentiment: np.ndarray,
    window: int,
    lag: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """
    Features: signature(level<=2) of path (S_t, P_{t-lag}) over last `window` points.
    Target: forward log return over `horizon` steps.
    """
    p = _clip01(price)
    s = np.asarray(sentiment, float)
    n = len(p)
    rows = []
    y = []
    t_out = []
    start = max(window + lag, 1)
    end = n - horizon
    for t in range(start, end):
        sl = slice(t - window, t + 1)
        s_path = s[sl]
        p_path = p[sl.start - lag : sl.stop - lag]
        if len(p_path) != len(s_path):
            continue
        path = np.column_stack([s_path, p_path])
        feats, _ = signature_features(path)
        rows.append(feats)
        y.append(math.log(p[t + horizon]) - math.log(p[t]))
        t_out.append(ts[t])
    X = np.vstack(rows) if rows else np.empty((0, 6))
    return X, np.asarray(y, float), t_out


def signed_area_from_sig_level2(sig: Signature) -> float:
    # 2D signed area = 0.5*(S12 - S21)
    return float(0.5 * (sig.level2[0, 1] - sig.level2[1, 0]))


def main() -> None:
    _ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="Gamma markets to scan")
    parser.add_argument("--dt_s", type=int, default=60, help="Resample seconds")
    parser.add_argument("--windows", type=str, default="60,180,360", help="Comma-separated window sizes (steps)")
    parser.add_argument("--lags", type=str, default="0,5,15,30,60", help="Comma-separated lag sizes (steps)")
    parser.add_argument("--horizon", type=int, default=30, help="Forward horizon (steps)")
    parser.add_argument("--train_frac", type=float, default=0.7, help="Train fraction")
    parser.add_argument("--lam", type=float, default=1e-2, help="Ridge lambda")
    parser.add_argument("--sentiment_csv", type=str, default="", help="CSV with columns ts,sentiment (or override cols below)")
    parser.add_argument("--sentiment_ts_col", type=str, default="ts")
    parser.add_argument("--sentiment_value_col", type=str, default="sentiment")
    parser.add_argument("--sentiment_align", type=str, default="ffill", help="ffill|nearest|zero")
    args = parser.parse_args()

    markets = fetch_gamma_markets(limit=args.limit)
    m = pick_market(markets)
    outcomes = parse_gamma_list_field(m.get("outcomes"))
    token_ids = parse_gamma_list_field(m.get("clobTokenIds"))
    idx = 0
    if outcomes[0].lower() == "no" and outcomes[1].lower() == "yes":
        idx = 1
    token = str(token_ids[idx])

    df = fetch_price_history(token, interval="max")
    df, _ = regularize_history(df, target_dt_s=args.dt_s)
    df = df.tail(60000)  # keep runtime bounded

    ts = df["ts"].to_numpy()
    price = df["price"].astype(float).to_numpy()
    if args.sentiment_csv:
        s_raw = load_sentiment_series(args.sentiment_csv, ts_col=args.sentiment_ts_col, value_col=args.sentiment_value_col)
        s_aligned = align_sentiment_to_index(s_raw, df["ts"], method=args.sentiment_align)
        sentiment = s_aligned.astype(float).to_numpy()
        # standardize to stabilize signatures/regression
        sentiment = (sentiment - float(np.mean(sentiment))) / (float(np.std(sentiment)) + 1e-12)
    else:
        sentiment = make_proxy_sentiment(price)

    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    lags = [int(x) for x in args.lags.split(",") if x.strip()]

    # Sweep windows/lags; score via out-of-sample IC and R^2
    rows = []
    for w in windows:
        for lag in lags:
            X, y, t_out = build_dataset(ts, price, sentiment, window=w, lag=lag, horizon=args.horizon)
            if len(y) < 500:
                continue
            n = len(y)
            ntr = int(args.train_frac * n)
            Xtr, ytr = X[:ntr], y[:ntr]
            Xte, yte = X[ntr:], y[ntr:]

            yhat, beta = ridge_fit_predict(Xtr, ytr, Xte, lam=args.lam)
            ic = corrcoef_safe(yhat, yte)
            mse = float(np.mean((yte - yhat) ** 2))
            r2 = float(1.0 - (np.sum((yte - yhat) ** 2) / (np.sum((yte - np.mean(yte)) ** 2) + 1e-12)))
            rows.append(
                {
                    "window": w,
                    "lag": lag,
                    "n": n,
                    "ic": ic,
                    "mse": mse,
                    "r2": r2,
                }
            )
    scores = pd.DataFrame(rows).sort_values(["window", "lag"])
    out_base = f"sig_{m.get('id')}_{token[:10]}"
    scores.to_csv(f"out/{out_base}_lag_window_scores.csv", index=False)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Heatmap: IC across lag/window
    if not scores.empty:
        piv = scores.pivot(index="window", columns="lag", values="ic")
        plt.figure(figsize=(9, 4))
        plt.imshow(piv.values, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Information Coefficient (corr(pred, fwd return))")
        plt.xticks(range(len(piv.columns)), [str(c) for c in piv.columns], rotation=0)
        plt.yticks(range(len(piv.index)), [str(i) for i in piv.index])
        plt.xlabel("Lag (steps)")
        plt.ylabel("Window (steps)")
        plt.title("Signature lead–lag sweep (proxy sentiment vs lagged price)")
        plt.tight_layout()
        plt.savefig(f"out/{out_base}_ic_heatmap.png", dpi=160)
        plt.close()

    # Fit best config and visualize "signature of disagreement" (signed area term)
    if not scores.empty:
        best = scores.sort_values("ic", ascending=False).iloc[0].to_dict()
        w = int(best["window"])
        lag = int(best["lag"])
        X, y, t_out = build_dataset(ts, price, sentiment, window=w, lag=lag, horizon=args.horizon)
        n = len(y)
        ntr = int(args.train_frac * n)
        Xtr, ytr, ttr = X[:ntr], y[:ntr], t_out[:ntr]
        Xte, yte, tte = X[ntr:], y[ntr:], t_out[ntr:]
        yhat, beta = ridge_fit_predict(Xtr, ytr, Xte, lam=args.lam)

        # Coef bar (exclude intercept)
        feat_names = {
            0: "L1_d0(ΔS)",
            1: "L1_d1(ΔP_lag)",
            2: "L2_d00",
            3: "L2_d01",
            4: "L2_d10",
            5: "L2_d11",
        }
        coef = beta[1:]
        order = np.argsort(np.abs(coef))[::-1]
        plt.figure(figsize=(8, 3))
        plt.bar([feat_names.get(int(i), str(i)) for i in order], coef[order])
        plt.xticks(rotation=30, ha="right")
        plt.title(f"Ridge coeffs (best IC) | window={w} lag={lag} horizon={args.horizon}")
        plt.tight_layout()
        plt.savefig(f"out/{out_base}_best_coeffs.png", dpi=160)
        plt.close()

        # Compute signed area signature term over windows and compare regimes
        areas = []
        agree = []
        for k in range(max(w + lag, 1), len(price) - args.horizon):
            sl = slice(k - w, k + 1)
            s_path = sentiment[sl]
            p_path = _clip01(price)[sl.start - lag : sl.stop - lag]
            if len(p_path) != len(s_path):
                continue
            path = np.column_stack([s_path, p_path])
            sig = signature_level2(path)
            a = signed_area_from_sig_level2(sig)
            areas.append(a)
            # agreement = sign(Δsentiment over window) matches sign(fwd return)
            ds = float(s_path[-1] - s_path[0])
            fr = float(math.log(_clip01(price)[k + args.horizon]) - math.log(_clip01(price)[k]))
            agree.append(1 if (ds == 0 or fr == 0 or math.copysign(1.0, ds) == math.copysign(1.0, fr)) else 0)

        areas = np.asarray(areas, float)
        agree = np.asarray(agree, int)
        a0 = areas[agree == 0]
        a1 = areas[agree == 1]
        plt.figure(figsize=(9, 3))
        bins = 60
        plt.hist(a1, bins=bins, alpha=0.6, label="agreement", density=True)
        plt.hist(a0, bins=bins, alpha=0.6, label="disagreement", density=True)
        plt.legend()
        plt.title("“Signature of disagreement”: signed area term distribution")
        plt.tight_layout()
        plt.savefig(f"out/{out_base}_signed_area_agree_disagree.png", dpi=160)
        plt.close()

        with open(f"out/{out_base}_best_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "market_id": m.get("id"),
                    "question": m.get("question"),
                    "token": token,
                    "best_window": w,
                    "best_lag": lag,
                    "best_ic": float(best["ic"]),
                    "best_r2": float(best["r2"]),
                    "horizon_steps": args.horizon,
                    "dt_s": args.dt_s,
                    "note": "Sentiment is a proxy from price momentum. Swap in real sentiment aligned to timestamps.",
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()

