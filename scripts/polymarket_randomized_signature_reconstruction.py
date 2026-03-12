import argparse
import json
import math
import os
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


def pick_market(markets: list[dict]) -> dict:
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
        return m
    raise RuntimeError("No suitable market found.")


def make_proxy_sentiment(prices: np.ndarray, span: int = 120) -> np.ndarray:
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


def sig_targets_level2(path: np.ndarray) -> np.ndarray:
    sig = signature_level2(path)
    return np.concatenate([sig.level1.reshape(-1), sig.level2.reshape(-1)])


def ridge_fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, lam: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-output ridge regression with intercept via augmentation.
    Returns (Y_pred, beta) where beta includes intercept in row 0.
    """
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    Ytr = np.asarray(Ytr, float)
    Xtr_aug = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
    Xte_aug = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
    I = np.eye(Xtr_aug.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.solve(Xtr_aug.T @ Xtr_aug + lam * I, Xtr_aug.T @ Ytr)
    Yhat = Xte_aug @ beta
    return Yhat, beta


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    ssr = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    return 1.0 - ssr / sst


class RandomVectorFields:
    def __init__(self, d: int, n: int, hidden: int, depth_two: bool, rng: np.random.Generator):
        self.d = d
        self.n = n
        self.hidden = hidden
        self.depth_two = depth_two
        self.rng = rng

        if depth_two:
            # V_i(y) = W2_i tanh(W1_i y + b1_i) + b2_i
            self.W1 = rng.normal(scale=1.0 / math.sqrt(n), size=(d, hidden, n))
            self.b1 = rng.normal(scale=0.1, size=(d, hidden))
            self.W2 = rng.normal(scale=1.0 / math.sqrt(hidden), size=(d, n, hidden))
            self.b2 = rng.normal(scale=0.1, size=(d, n))
        else:
            # V_i(y) = tanh(A_i y + b_i)
            self.A = rng.normal(scale=1.0 / math.sqrt(n), size=(d, n, n))
            self.b = rng.normal(scale=0.1, size=(d, n))

    def V(self, i: int, y: np.ndarray) -> np.ndarray:
        if self.depth_two:
            h = np.tanh(self.W1[i] @ y + self.b1[i])
            return self.W2[i] @ h + self.b2[i]
        return np.tanh(self.A[i] @ y + self.b[i])


def cde_terminal_state(
    X: np.ndarray,
    vfields: RandomVectorFields,
    y0: np.ndarray,
    vf_scale: float,
) -> np.ndarray:
    """
    Euler discretization of the CDE for bounded-variation piecewise-linear controls:
      Y_{k+1} = Y_k + sum_i V_i(Y_k) * ΔX_k^i
    """
    y = np.asarray(y0, float).copy()
    dX = np.diff(X, axis=0)
    for dx in dX:
        inc = np.zeros_like(y)
        for i in range(vfields.d):
            inc += (vf_scale * vfields.V(i, y)) * float(dx[i])
        y = y + inc
    return y


def normalize_path(X: np.ndarray) -> np.ndarray:
    """
    Simple per-window normalization to keep the CDE numerically stable and make reconstruction well-conditioned.
    (Still consistent with the paper: we're just rescaling the input control path.)
    """
    X = np.asarray(X, float)
    X0 = X - X[:1]
    scale = np.std(X0, axis=0) + 1e-8
    return X0 / scale


class RandomizedSignatureExtractor:
    """
    Fixed (random) vector fields + fixed initial values y_j, as in the paper’s randomized signature setup.
    """

    def __init__(self, d: int, n: int, m_init: int, seed: int, depth_two: bool, vf_scale: float):
        self.d = d
        self.n = n
        self.m_init = m_init
        self.seed = seed
        self.depth_two = depth_two
        self.vf_scale = float(vf_scale)

        rng = np.random.default_rng(seed)
        self.vfields = RandomVectorFields(d=d, n=n, hidden=max(4, n), depth_two=depth_two, rng=rng)
        self.y0s = [rng.normal(scale=1.0, size=(n,)) for _ in range(m_init)]

    def features(self, X: np.ndarray) -> np.ndarray:
        Xn = normalize_path(X)
        ys = [cde_terminal_state(Xn, self.vfields, y0, vf_scale=self.vf_scale) for y0 in self.y0s]
        return np.concatenate(ys)


def build_windows(
    s: np.ndarray,
    p: np.ndarray,
    window: int,
    lag: int,
    step: int = 5,
) -> list[np.ndarray]:
    p = _clip01(p)
    n = len(p)
    Xs = []
    start = window + lag
    for t in range(start, n, step):
        sl = slice(t - window, t + 1)
        s_path = s[sl]
        p_path = p[sl.start - lag : sl.stop - lag]
        if len(p_path) != len(s_path):
            continue
        Xs.append(np.column_stack([s_path, p_path]))
    return Xs


def main() -> None:
    _ensure_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--dt_s", type=int, default=60)
    ap.add_argument("--window", type=int, default=360, help="Window length (steps)")
    ap.add_argument("--lag", type=int, default=60, help="Lag on price channel (steps)")
    ap.add_argument("--step", type=int, default=10, help="Stride between windows")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--m_init", type=int, default=8, help="Number of initial values y_j")
    ap.add_argument("--depth_two", action="store_true", help="Use depth-two neural vector fields")
    ap.add_argument("--n_list", type=str, default="8,16,32", help="Hidden dimensions to test")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--vf_scale", type=float, default=0.05, help="Scale applied to vector field outputs for stability")
    ap.add_argument("--sentiment_csv", type=str, default="", help="CSV with columns ts,sentiment (or override cols below)")
    ap.add_argument("--sentiment_ts_col", type=str, default="ts")
    ap.add_argument("--sentiment_value_col", type=str, default="sentiment")
    ap.add_argument("--sentiment_align", type=str, default="ffill", help="ffill|nearest|zero")
    args = ap.parse_args()

    markets = fetch_gamma_markets(limit=args.limit)
    m = pick_market(markets)
    outcomes = parse_gamma_list_field(m.get("outcomes"))
    token_ids = parse_gamma_list_field(m.get("clobTokenIds"))
    idx = 0
    if len(outcomes) == 2 and outcomes[0].lower() == "no" and outcomes[1].lower() == "yes":
        idx = 1
    token = str(token_ids[idx])

    df = fetch_price_history(token, interval="max")
    df, _ = regularize_history(df, target_dt_s=args.dt_s)
    df = df.tail(60000)
    price = df["price"].astype(float).to_numpy()
    if args.sentiment_csv:
        s_raw = load_sentiment_series(args.sentiment_csv, ts_col=args.sentiment_ts_col, value_col=args.sentiment_value_col)
        s_aligned = align_sentiment_to_index(s_raw, df["ts"], method=args.sentiment_align)
        sent = s_aligned.astype(float).to_numpy()
        sent = (sent - float(np.mean(sent))) / (float(np.std(sent)) + 1e-12)
    else:
        sent = make_proxy_sentiment(price)

    X_windows = build_windows(sent, price, window=args.window, lag=args.lag, step=args.step)
    if len(X_windows) < 200:
        raise SystemExit(f"Not enough windows ({len(X_windows)}) — try smaller window/lag/step.")

    # Targets: true signature (level<=2) for each window
    # IMPORTANT: compute targets on the *same normalized path* we feed to the CDE, for a coherent reconstruction task.
    Y = np.vstack([sig_targets_level2(normalize_path(X)) for X in X_windows])

    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    rows = []

    # Train/test split on windows
    n_total = len(X_windows)
    ntr = int(args.train_frac * n_total)
    Xtr_windows = X_windows[:ntr]
    Xte_windows = X_windows[ntr:]
    Ytr = Y[:ntr]
    Yte = Y[ntr:]

    for n in n_list:
        extractor = RandomizedSignatureExtractor(
            d=2, n=n, m_init=args.m_init, seed=args.seed, depth_two=args.depth_two, vf_scale=args.vf_scale
        )
        # Build randomized-signature features Φ(X) for each window
        Phi_tr = np.vstack([extractor.features(X) for X in Xtr_windows])
        Phi_te = np.vstack([extractor.features(X) for X in Xte_windows])

        Yhat, _ = ridge_fit_predict(Phi_tr, Ytr, Phi_te, lam=args.lam)
        # component-wise R^2
        for j in range(Yte.shape[1]):
            rows.append(
                {
                    "n": n,
                    "component": j,
                    "r2": r2_score(Yte[:, j], Yhat[:, j]),
                }
            )
        rows.append({"n": n, "component": "mean", "r2": float(np.mean([r2_score(Yte[:, j], Yhat[:, j]) for j in range(Yte.shape[1])]))})

    res = pd.DataFrame(rows)
    tag = "depth2" if args.depth_two else "depth1"
    out_base = f"rand_sig_recon_{m.get('id')}_{tag}_w{args.window}_lag{args.lag}_m{args.m_init}"
    res.to_csv(f"out/{out_base}_r2.csv", index=False)

    # Plot R^2 vs hidden dimension
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    mean_df = res[res["component"] == "mean"].sort_values("n")
    plt.figure(figsize=(7, 3.2))
    plt.plot(mean_df["n"], mean_df["r2"], marker="o")
    plt.xlabel("Hidden dimension N")
    plt.ylabel("Mean R² (reconstruct level≤2 signature)")
    plt.title("Signature reconstruction from randomized signatures")
    plt.tight_layout()
    plt.savefig(f"out/{out_base}_mean_r2.png", dpi=160)
    plt.close()

    # Save metadata
    with open(f"out/{out_base}_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "paper": "Signature Reconstruction from Randomized Signatures (Glückstad, Muça Cirone, Teichmann)",
                "market_id": m.get("id"),
                "question": m.get("question"),
                "token": token,
                "dt_s": args.dt_s,
                "window": args.window,
                "lag": args.lag,
                "step": args.step,
                "m_init": args.m_init,
                "depth_two": bool(args.depth_two),
                "n_list": n_list,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

