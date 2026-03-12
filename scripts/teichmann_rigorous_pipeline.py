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
    os.makedirs("data/api", exist_ok=True)
    mpl_cfg = os.path.abspath(".mplconfig")
    os.makedirs(mpl_cfg, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)
    os.environ.setdefault("MPLBACKEND", "Agg")


def _json_get(url: str, params: dict | None = None, timeout_s: int = 30) -> dict | list:
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    r = requests.get(url, params=params, timeout=timeout_s, verify=False)
    r.raise_for_status()
    return r.json()


def _clip01(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def fetch_gamma_market_by_id(market_id: str) -> dict:
    """
    Gamma supports filtering by id via /markets?id=...
    """
    markets = _json_get("https://gamma-api.polymarket.com/markets", params={"id": str(market_id)})
    if isinstance(markets, list) and markets:
        return markets[0]
    raise RuntimeError(f"Market id {market_id} not found via Gamma id filter.")


def _parse_gamma_list_field(x) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def pick_market_with_overlap(
    sentiment_start: pd.Timestamp,
    sentiment_end: pd.Timestamp,
    max_candidates: int = 40,
) -> tuple[dict, str, str]:
    """
    Choose a 2-outcome market whose CLOB price history covers [sentiment_start, sentiment_end].
    Returns (gamma_market, token_id, outcome_name).
    """
    # Use open markets list to get recent ids (Gamma list without filters tends to return very old markets).
    markets = _json_get("https://gamma-api.polymarket.com/markets", params={"closed": "false", "limit": "500"})
    if not isinstance(markets, list):
        raise RuntimeError("Unexpected Gamma response while picking market.")

    def vol(m: dict) -> float:
        try:
            return float(m.get("volumeNum") or 0.0)
        except Exception:
            return 0.0

    # Coarse filter by start/end dates if present, then sort by liquidity/volume.
    cand = []
    for m in markets:
        sd = m.get("startDate")
        ed = m.get("endDate")
        try:
            sd = pd.to_datetime(sd, utc=True) if sd else None
            ed = pd.to_datetime(ed, utc=True) if ed else None
        except Exception:
            sd = None
            ed = None
        if sd is not None and sd > sentiment_end:
            continue
        if ed is not None and ed < sentiment_start:
            continue
        outcomes = _parse_gamma_list_field(m.get("outcomes"))
        tids = _parse_gamma_list_field(m.get("clobTokenIds"))
        if len(outcomes) != 2 or len(tids) != 2:
            continue
        cand.append(m)

    cand = sorted(cand, key=vol, reverse=True)[:max_candidates]
    buf = pd.Timedelta(days=1)
    start = sentiment_start - buf
    end = sentiment_end + buf

    best = None
    best_score = -1.0
    for m in cand:
        outcomes = _parse_gamma_list_field(m.get("outcomes"))
        tids = _parse_gamma_list_field(m.get("clobTokenIds"))
        # prefer YES if present
        idx = 0
        if outcomes[0].lower() == "no" and outcomes[1].lower() == "yes":
            idx = 1
        elif outcomes[0].lower() == "yes":
            idx = 0
        elif outcomes[1].lower() == "yes":
            idx = 1
        token_id = str(tids[idx])
        try:
            dfp = fetch_clob_prices_history(token_id, interval="max", start_ts=start, end_ts=end)
        except Exception:
            continue
        if dfp.empty:
            continue
        if not (dfp["ts"].min() <= sentiment_start and dfp["ts"].max() >= sentiment_end):
            continue
        # Prefer markets with actual movement in this window.
        score = float(np.nanstd(dfp["price"].astype(float).to_numpy()))
        if score > best_score:
            best_score = score
            best = (m, token_id, str(outcomes[idx]))

    if best is None:
        raise RuntimeError("Could not find a market whose price history overlaps the sentiment range in the scanned set.")
    return best


def fetch_clob_prices_history(
    token_id: str,
    interval: str = "max",
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    params: dict[str, str] = {"market": token_id, "interval": interval}
    if start_ts is not None:
        params["startTs"] = str(int(pd.Timestamp(start_ts).timestamp()))
    if end_ts is not None:
        params["endTs"] = str(int(pd.Timestamp(end_ts).timestamp()))
    try:
        j = _json_get("https://clob.polymarket.com/prices-history", params=params)
    except Exception:
        # Some tokens reject bounded queries with 400. Fallback to full history and slice later.
        j = _json_get("https://clob.polymarket.com/prices-history", params={"market": token_id, "interval": interval})
    hist = j.get("history", [])
    df = pd.DataFrame(hist)
    if df.empty:
        return df
    df = df.rename(columns={"t": "ts", "p": "price"})
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ts", "price"]).sort_values("ts").drop_duplicates("ts")
    if start_ts is not None:
        start_utc = pd.to_datetime(start_ts, utc=True)
        df = df[df["ts"] >= start_utc]
    if end_ts is not None:
        end_utc = pd.to_datetime(end_ts, utc=True)
        df = df[df["ts"] <= end_utc]
    return df[["ts", "price"]]


def regularize_prices(df: pd.DataFrame, dt_s: int) -> pd.DataFrame:
    s = df.set_index("ts")["price"].astype(float)
    s = s.resample(f"{dt_s}s").last().ffill()
    out = s.reset_index().rename(columns={"index": "ts", "price": "price"})
    out = out.dropna()
    return out


def normalize_path(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    X0 = X - X[:1]
    scale = np.std(X0, axis=0) + 1e-8
    return X0 / scale


def ridge_fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    Ytr = np.asarray(Ytr, float)
    Xtr_aug = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
    Xte_aug = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
    I = np.eye(Xtr_aug.shape[1])
    I[0, 0] = 0.0  # don't penalize intercept
    beta = np.linalg.solve(Xtr_aug.T @ Xtr_aug + lam * I, Xtr_aug.T @ Ytr)
    return Xte_aug @ beta, beta


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    ssr = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - float(np.mean(y_true))) ** 2) + 1e-12)
    return 1.0 - ssr / sst


def r2_score_or_nan(y_true: np.ndarray, y_pred: np.ndarray, var_eps: float = 1e-10) -> float:
    """
    R² is ill-conditioned when y_true has ~0 variance. Return NaN in that case.
    """
    y_true = np.asarray(y_true, float).reshape(-1)
    if float(np.var(y_true)) < var_eps:
        return float("nan")
    return r2_score(y_true, y_pred)


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    if len(a) < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def truncated_signature_increments(dx: np.ndarray, m: int) -> list[np.ndarray]:
    """
    Segment signature for one increment v = dx under piecewise-linear BV path:
      Sig(v) = exp(v) truncated => level k tensor = v^{⊗k}/k!
    Returns list levels[0..m] where levels[0] is scalar 1.0.
    """
    d = dx.shape[0]
    levels: list[np.ndarray] = [np.array(1.0)]
    # level 1
    levels.append(dx.copy())
    for k in range(2, m + 1):
        # build outer product iteratively: v^{⊗k}
        t = dx
        for _ in range(k - 1):
            t = np.tensordot(t, dx, axes=0)
        levels.append(t / math.factorial(k))
    return levels


def chen_concat(A: list[np.ndarray], B: list[np.ndarray], m: int) -> list[np.ndarray]:
    """
    Truncated tensor algebra concatenation:
      (A ⊗ B)_k = sum_{i=0..k} A_i ⊗ B_{k-i}
    where tensor product is outer product.
    """
    out: list[np.ndarray] = []
    out.append(np.array(1.0))
    for k in range(1, m + 1):
        acc = None
        for i in range(0, k + 1):
            Ai = A[i]
            Bj = B[k - i]
            term = Ai if i == 0 else np.tensordot(Ai, Bj, axes=0)
            if i == 0:
                term = Bj
            elif k - i == 0:
                term = Ai
            else:
                term = np.tensordot(Ai, Bj, axes=0)
            acc = term if acc is None else (acc + term)
        out.append(acc if acc is not None else np.zeros((A[1].shape[0],) * k))
    return out


def signature_truncated(path: np.ndarray, m: int) -> list[np.ndarray]:
    """
    Truncated signature up to order m for a piecewise-linear path.
    Returns levels[0..m], levels[0]=1 scalar.
    """
    x = np.asarray(path, float)
    if x.ndim != 2 or x.shape[0] < 2:
        d = x.shape[1] if x.ndim == 2 else 0
        out = [np.array(1.0)]
        for k in range(1, m + 1):
            out.append(np.zeros((d,) * k, dtype=float))
        return out

    dX = np.diff(x, axis=0)
    # identity signature
    A: list[np.ndarray] = [np.array(1.0)]
    for k in range(1, m + 1):
        A.append(np.zeros((x.shape[1],) * k, dtype=float))
    for v in dX:
        B = truncated_signature_increments(v, m)
        A = chen_concat(A, B, m)
    return A


def signature_targets(path: np.ndarray, m: int) -> np.ndarray:
    levels = signature_truncated(path, m)
    feats = []
    for k in range(1, m + 1):
        feats.append(levels[k].reshape(-1))
    return np.concatenate(feats)


class RandomVectorFields:
    def __init__(self, d: int, n: int, hidden: int, depth_two: bool, activation: str, rng: np.random.Generator):
        self.d = d
        self.n = n
        self.hidden = hidden
        self.depth_two = depth_two
        self.activation = activation
        self.rng = rng
        if depth_two:
            if activation == "exp":
                # Depth-two nested exponential fields aligned with Teichmann setup:
                # V_i(y) = exp(A_i exp(D_i y))
                self.A = rng.normal(scale=1.0 / math.sqrt(n), size=(d, n, n))
                self.D = rng.normal(scale=1.0 / math.sqrt(n), size=(d, n))
            else:
                self.W1 = rng.normal(scale=1.0 / math.sqrt(n), size=(d, hidden, n))
                self.b1 = rng.normal(scale=0.1, size=(d, hidden))
                self.W2 = rng.normal(scale=1.0 / math.sqrt(hidden), size=(d, n, hidden))
                self.b2 = rng.normal(scale=0.1, size=(d, n))
        else:
            self.A = rng.normal(scale=1.0 / math.sqrt(n), size=(d, n, n))
            self.b = rng.normal(scale=0.1, size=(d, n))

    def V(self, i: int, y: np.ndarray) -> np.ndarray:
        if self.depth_two:
            if self.activation == "exp":
                # Numerical stability: clip arguments before exponentials.
                z1 = np.clip(self.D[i] * y, -20.0, 20.0)
                h = np.exp(z1)
                z2 = np.clip(self.A[i] @ h, -20.0, 20.0)
                return np.exp(z2)
            h = np.tanh(self.W1[i] @ y + self.b1[i])
            return self.W2[i] @ h + self.b2[i]
        return np.tanh(self.A[i] @ y + self.b[i])


def cde_terminal_state(X: np.ndarray, vfields: RandomVectorFields, y0: np.ndarray, vf_scale: float) -> np.ndarray:
    """
    Euler discretization of the CDE on a BV path:
      Y_{k+1} = Y_k + Σ_i V_i(Y_k) ΔX_k^i
    """
    y = np.asarray(y0, float).copy()
    dX = np.diff(X, axis=0)
    for dx in dX:
        inc = np.zeros_like(y)
        for i in range(vfields.d):
            inc += (vf_scale * vfields.V(i, y)) * float(dx[i])
        y = y + inc
    return y


class RandomizedSignatureExtractor:
    def __init__(self, d: int, n: int, m_init: int, seed: int, depth_two: bool, vf_scale: float, activation: str):
        self.d = d
        self.n = n
        self.m_init = m_init
        self.seed = seed
        self.depth_two = depth_two
        self.vf_scale = float(vf_scale)
        rng = np.random.default_rng(seed)
        self.vfields = RandomVectorFields(
            d=d,
            n=n,
            hidden=max(4, n),
            depth_two=depth_two,
            activation=activation,
            rng=rng,
        )
        self.y0s = [rng.normal(scale=1.0, size=(n,)) for _ in range(m_init)]

    def features(self, X: np.ndarray) -> np.ndarray:
        Xn = normalize_path(X)
        ys = [cde_terminal_state(Xn, self.vfields, y0, vf_scale=self.vf_scale) for y0 in self.y0s]
        return np.concatenate(ys)


@dataclass
class WindowDataset:
    X_windows: list[np.ndarray]
    Y_sig: np.ndarray
    y_fwdret: np.ndarray


def build_windows(s: np.ndarray, p: np.ndarray, window: int, lag: int, horizon: int, step: int, sig_order: int) -> WindowDataset:
    p = _clip01(np.asarray(p, float))
    s = np.asarray(s, float)
    n = len(p)
    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    rets: list[float] = []
    start = window + lag
    end = n - horizon
    for t in range(start, end, step):
        sl = slice(t - window, t + 1)
        s_path = s[sl]
        p_path = p[sl.start - lag : sl.stop - lag]
        if len(p_path) != len(s_path):
            continue
        X = np.column_stack([s_path, p_path])
        Xs.append(X)
        # targets computed on normalized path for coherence
        Ys.append(signature_targets(normalize_path(X), sig_order))
        rets.append(float(math.log(p[t + horizon]) - math.log(p[t])))
    Y = np.vstack(Ys) if Ys else np.empty((0, 0))
    return WindowDataset(X_windows=Xs, Y_sig=Y, y_fwdret=np.asarray(rets, float))


def main() -> None:
    _ensure_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--market_id", type=str, default="", help="Gamma market id to fetch (optional if auto_pick_market)")
    ap.add_argument("--auto_pick_market", action="store_true", help="Pick a market whose price history overlaps sentiment range")
    ap.add_argument("--sentiment_csv", type=str, required=True)
    ap.add_argument("--sentiment_ts_col", type=str, default="ts")
    ap.add_argument("--sentiment_value_col", type=str, default="sentiment")
    ap.add_argument("--sentiment_align", type=str, default="ffill")
    ap.add_argument("--dt_s", type=int, default=60)
    ap.add_argument("--min_points", type=int, default=1000, help="Minimum overlapping regularized price points required")
    ap.add_argument(
        "--price_source",
        type=str,
        default="history",
        choices=["history", "trades"],
        help="Price source: 'history' uses public /prices-history; 'trades' uses authenticated /data/trades (requires POLY_* env vars).",
    )
    ap.add_argument("--window", type=int, default=360)
    ap.add_argument("--lag", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--sig_order", type=int, default=3, help="Truncated signature order to reconstruct (<=3 recommended)")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--vf_scale", type=float, default=0.02)
    ap.add_argument("--activation", type=str, default="exp", choices=["exp", "tanh"])
    ap.add_argument("--m_init", type=int, default=60, help="Number of initial states y_j")
    ap.add_argument("--n_list", type=str, default="16,32", help="Hidden dims N")
    ap.add_argument("--depth_two", action="store_true")
    ap.add_argument("--n_draws", type=int, default=5, help="Independent random draws of vector fields")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_prefix", type=str, default="teichmann_rigorous")
    args = ap.parse_args()
    if args.depth_two and args.activation == "exp":
        min_n = min(int(x) for x in args.n_list.split(",") if x.strip())
        if min_n < (args.sig_order - 1):
            raise SystemExit(
                f"Teichmann depth-two setting requires N >= sig_order-1. Got min N={min_n}, sig_order={args.sig_order}."
            )

    # 1) Load sentiment first to get its time range (for overlap-aware market selection)
    s_raw = load_sentiment_series(args.sentiment_csv, ts_col=args.sentiment_ts_col, value_col=args.sentiment_value_col)
    sentiment_start = s_raw.index.min()
    sentiment_end = s_raw.index.max()

    # 2) Fetch or pick market + token id (YES if possible)
    if args.auto_pick_market:
        m, token_id, outcome_name = pick_market_with_overlap(sentiment_start, sentiment_end)
        market_id = str(m.get("id"))
    else:
        if not args.market_id:
            raise SystemExit("Provide --market_id or use --auto_pick_market")
        market_id = str(args.market_id)
        m = fetch_gamma_market_by_id(market_id)
        outcomes = _parse_gamma_list_field(m.get("outcomes"))
        tids = _parse_gamma_list_field(m.get("clobTokenIds"))
        if not (len(outcomes) == 2 and len(tids) == 2):
            raise SystemExit("Market does not look like a simple 2-outcome CLOB market.")
        idx = 0
        if outcomes[0].lower() == "no" and outcomes[1].lower() == "yes":
            idx = 1
        elif outcomes[0].lower() == "yes":
            idx = 0
        elif outcomes[1].lower() == "yes":
            idx = 1
        token_id = str(tids[idx])
        outcome_name = str(outcomes[idx])

    with open(f"data/api/gamma_market_{market_id}.json", "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

    # 3) Fetch real price history
    # Pull only the overlapping history slice directly from the API to avoid gigantic resamples.
    buf = pd.Timedelta(days=1)
    if args.price_source == "trades":
        from clob_l2_auth import load_l2_creds_from_env
        from clob_trades import fetch_market_trades, trades_to_price_df

        creds = load_l2_creds_from_env("POLY_")
        after_ts = int((sentiment_start - buf).timestamp())
        before_ts = int((sentiment_end + buf).timestamp())
        trades = fetch_market_trades(creds, token_id, after_ts=after_ts, before_ts=before_ts)
        dfp = trades_to_price_df(trades)
    else:
        dfp = fetch_clob_prices_history(
            token_id,
            interval="max",
            start_ts=sentiment_start - buf,
            end_ts=sentiment_end + buf,
        )

    if dfp.empty:
        raise SystemExit("No price history returned (try --price_source trades with POLY_* env vars).")
    dfp = regularize_prices(dfp, dt_s=args.dt_s)
    dfp = dfp[(dfp["ts"] >= sentiment_start - buf) & (dfp["ts"] <= sentiment_end + buf)].copy()
    if len(dfp) < args.min_points:
        raise SystemExit(
            f"Not enough overlapping price points after slicing to sentiment range. "
            f"Have {len(dfp)}, need at least {args.min_points}."
        )
    dfp.to_csv(f"data/api/clob_prices_{token_id}_dt{args.dt_s}s.csv", index=False)

    # 4) Align sentiment to price timestamps
    s_al = align_sentiment_to_index(s_raw, dfp["ts"], method=args.sentiment_align)
    s = s_al.astype(float).to_numpy()
    # standardize sentiment
    s = (s - float(np.mean(s))) / (float(np.std(s)) + 1e-12)

    p = dfp["price"].astype(float).to_numpy()

    # 4) Build window dataset (path, signature targets, forward return)
    ds = build_windows(s, p, window=args.window, lag=args.lag, horizon=args.horizon, step=args.step, sig_order=args.sig_order)
    if len(ds.X_windows) < 200:
        raise SystemExit(f"Not enough windows built: {len(ds.X_windows)}. Try smaller window/lag/step.")

    n_total = len(ds.X_windows)
    ntr = int(args.train_frac * n_total)
    Xtr, Xte = ds.X_windows[:ntr], ds.X_windows[ntr:]
    Ytr_sig, Yte_sig = ds.Y_sig[:ntr], ds.Y_sig[ntr:]
    ytr_ret, yte_ret = ds.y_fwdret[:ntr], ds.y_fwdret[ntr:]

    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]

    rows_recon = []
    rows_pred = []

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

                # 5a) Reconstruction: Phi(X) -> Sig^{<=m}(X)
                Yhat_sig, _ = ridge_fit_predict(Phi_tr, Ytr_sig, Phi_te, lam=args.lam)
                comp_r2 = [r2_score_or_nan(Yte_sig[:, j], Yhat_sig[:, j]) for j in range(Yte_sig.shape[1])]
                comp_r2_f = [x for x in comp_r2 if np.isfinite(x)]
                rows_recon.append(
                    {
                        "depth_two": bool(depth_tag),
                        "N": n,
                        "draw": draw,
                        "mean_r2": float(np.mean(comp_r2_f)) if comp_r2_f else float("nan"),
                        "median_r2": float(np.median(comp_r2_f)) if comp_r2_f else float("nan"),
                        "min_r2": float(np.min(comp_r2_f)) if comp_r2_f else float("nan"),
                        "max_r2": float(np.max(comp_r2_f)) if comp_r2_f else float("nan"),
                    }
                )
                for j, r2 in enumerate(comp_r2):
                    rows_recon.append({"depth_two": bool(depth_tag), "N": n, "draw": draw, "component": j, "r2": float(r2)})

                # 5b) Predictive: Phi(X) -> forward return (optional metric)
                yhat_ret, _ = ridge_fit_predict(Phi_tr, ytr_ret.reshape(-1, 1), Phi_te, lam=args.lam)
                yhat_ret = yhat_ret.reshape(-1)
                rows_pred.append(
                    {
                        "depth_two": bool(depth_tag),
                        "N": n,
                        "draw": draw,
                        "ic": corrcoef_safe(yhat_ret, yte_ret),
                        "r2": r2_score_or_nan(yte_ret, yhat_ret),
                        "mse": float(np.mean((yte_ret - yhat_ret) ** 2)),
                    }
                )

    recon = pd.DataFrame(rows_recon)
    pred = pd.DataFrame(rows_pred)

    tag = "depth1+2" if args.depth_two else "depth1"
    base = f"{args.out_prefix}_mkt{market_id}_tok{token_id[:10]}_{tag}_sig{args.sig_order}"
    recon.to_csv(f"out/{base}_reconstruction_metrics.csv", index=False)
    pred.to_csv(f"out/{base}_prediction_metrics.csv", index=False)

    # 6) Plots
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def plot_mean_r2(depth_two: bool) -> None:
        df = recon[(recon.get("component").isna()) & (recon["depth_two"] == depth_two)].copy()
        if df.empty:
            return
        g = df.groupby("N")["mean_r2"]
        x = np.array(sorted(g.mean().index))
        y = np.array([g.mean().loc[i] for i in x])
        yerr = np.array([g.std(ddof=1).loc[i] if i in g.std(ddof=1).index else 0.0 for i in x])
        plt.figure(figsize=(7, 3.2))
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
        plt.xlabel("Hidden dimension N")
        plt.ylabel("Mean R² (Sig reconstruction)")
        plt.title(f"Teichmann reconstruction (depth_two={depth_two})")
        plt.tight_layout()
        plt.savefig(f"out/{base}_mean_r2_depth2_{int(depth_two)}.png", dpi=160)
        plt.close()

    plot_mean_r2(False)
    if args.depth_two:
        plot_mean_r2(True)

    # Prediction IC vs N
    for depth_two in sorted(pred["depth_two"].unique().tolist()):
        df = pred[pred["depth_two"] == depth_two].copy()
        g = df.groupby("N")["ic"]
        x = np.array(sorted(g.mean().index))
        y = np.array([g.mean().loc[i] for i in x])
        yerr = np.array([g.std(ddof=1).loc[i] if i in g.std(ddof=1).index else 0.0 for i in x])
        plt.figure(figsize=(7, 3.2))
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
        plt.xlabel("Hidden dimension N")
        plt.ylabel("IC (corr(pred, fwd return))")
        plt.title(f"Forward-return predictability from randomized signatures (depth_two={depth_two})")
        plt.tight_layout()
        plt.savefig(f"out/{base}_ic_depth2_{int(depth_two)}.png", dpi=160)
        plt.close()

    # Save run metadata
    with open(f"out/{base}_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "paper": "Signature Reconstruction from Randomized Signatures (Glückstad, Muça Cirone, Teichmann)",
                "market_id": market_id,
                "question": m.get("question"),
                "token_id": token_id,
                "outcome": outcome_name,
                "dt_s": args.dt_s,
                "window": args.window,
                "lag": args.lag,
                "horizon": args.horizon,
                "step": args.step,
                "sig_order": args.sig_order,
                "m_init": args.m_init,
                "n_list": n_list,
                "depth_two": bool(args.depth_two),
                "n_draws": args.n_draws,
                "lam": args.lam,
                "vf_scale": args.vf_scale,
                "activation": args.activation,
                "sentiment_csv": os.path.abspath(args.sentiment_csv),
                "sentiment_align": args.sentiment_align,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

