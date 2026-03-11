import os
import math
import time
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize


def _ensure_dirs() -> None:
    os.makedirs("out", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    # Avoid matplotlib trying to write to ~/.matplotlib (often unwritable in sandboxes)
    mpl_cfg = os.path.abspath(".mplconfig")
    os.makedirs(mpl_cfg, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)
    # Force a headless backend (the default macOS backend aborts in sandboxed/non-GUI runs).
    os.environ.setdefault("MPLBACKEND", "Agg")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def _clip01(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _json_get(url: str, params: dict | None = None, timeout_s: int = 30) -> dict | list:
    # CLOB/Gamma endpoints are public; sandbox often lacks CA roots so we disable verify.
    # This is fine for a data experiment; do NOT do this for real trading infra.
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    r = requests.get(url, params=params, timeout=timeout_s, verify=False)
    r.raise_for_status()
    return r.json()


def fetch_gamma_markets(limit: int = 100) -> list[dict]:
    # Keep it simple: fetch some open markets and later filter by having enough history.
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


def regularize_history(df: pd.DataFrame, target_dt_s: int | None = None) -> tuple[pd.DataFrame, float]:
    """
    Convert irregular ticks into a regular time series with forward-fill.
    Returns (regular_df, dt_seconds).
    """
    if df.empty:
        return df, float("nan")

    s = df.set_index("ts")["price"].astype(float)
    diffs = s.index.to_series().diff().dt.total_seconds().dropna()
    dt_s = float(np.nanmedian(diffs.values)) if len(diffs) else 60.0
    dt_s = float(max(1.0, min(dt_s, 3600.0)))
    if target_dt_s is not None:
        dt_s = float(target_dt_s)

    rule = f"{int(dt_s)}s"
    s_reg = s.resample(rule).last().ffill()
    out = s_reg.reset_index().rename(columns={"index": "ts", "price": "price"})
    out = out.dropna()
    return out, dt_s


@dataclass
class FitResult:
    model: str
    params: dict
    test_loglik: float
    test_mse_p: float
    n_train: int
    n_test: int


class LogitBrownian:
    name = "logit_brownian"

    def fit(self, p: np.ndarray, dt: float) -> dict:
        x = _logit(_clip01(p))
        dx = np.diff(x)
        mu = float(dx.mean() / dt)
        sigma = float(dx.std(ddof=1) / math.sqrt(dt)) if len(dx) > 1 else 0.0
        return {"mu": mu, "sigma": max(sigma, 1e-12)}

    def one_step_metrics(self, p: np.ndarray, dt: float, params: dict) -> tuple[float, float]:
        x = _logit(_clip01(p))
        mu, sigma = params["mu"], params["sigma"]
        m = x[:-1] + mu * dt
        v = (sigma**2) * dt
        ll = _gaussian_loglik(x[1:], m, v)
        p_pred = _sigmoid(m)
        mse = float(np.mean((p[1:] - p_pred) ** 2))
        return ll, mse


class LogitOU:
    name = "logit_ou"

    def fit(self, p: np.ndarray, dt: float) -> dict:
        x = _logit(_clip01(p))
        x0, x1 = x[:-1], x[1:]
        if len(x0) < 5:
            return {"mu": float(np.mean(x)), "b": 0.999, "s2": 1e-6}

        b = float(np.cov(x0, x1, ddof=1)[0, 1] / np.var(x0, ddof=1))
        b = float(np.clip(b, 1e-6, 0.999999))  # enforce OU-like mean reversion
        a = float(x1.mean() - b * x0.mean())
        mu = float(a / (1.0 - b))

        resid = x1 - (a + b * x0)
        s2 = float(np.var(resid, ddof=1))
        s2 = max(s2, 1e-12)
        return {"mu": mu, "b": b, "a": a, "s2": s2}

    def one_step_metrics(self, p: np.ndarray, dt: float, params: dict) -> tuple[float, float]:
        x = _logit(_clip01(p))
        a, b, s2 = params["a"], params["b"], params["s2"]
        m = a + b * x[:-1]
        ll = _gaussian_loglik(x[1:], m, s2)
        p_pred = _sigmoid(m)
        mse = float(np.mean((p[1:] - p_pred) ** 2))
        return ll, mse


class JacobiEuler:
    """
    Jacobi / Wright–Fisher-like diffusion on price/probability p in (0,1):
      dP = kappa (theta - P) dt + sigma * sqrt(P(1-P)) dW

    We fit a Gaussian Euler quasi-likelihood:
      P_{t+dt} ~ Normal(P_t + kappa (theta - P_t) dt, sigma^2 P_t(1-P_t) dt)
    """

    name = "jacobi_euler"

    def fit(self, p: np.ndarray, dt: float) -> dict:
        p = _clip01(p)
        p0, p1 = p[:-1], p[1:]

        def unpack(z: np.ndarray) -> tuple[float, float, float]:
            kappa = float(np.exp(z[0]))
            theta = float(1.0 / (1.0 + np.exp(-z[1])))
            sigma = float(np.exp(z[2]))
            return kappa, theta, sigma

        def nll(z: np.ndarray) -> float:
            kappa, theta, sigma = unpack(z)
            m = p0 + kappa * (theta - p0) * dt
            v = (sigma**2) * p0 * (1.0 - p0) * dt
            v = np.maximum(v, 1e-12)
            return float(-_gaussian_loglik(p1, m, v))

        # crude init from moments
        z0 = np.array([math.log(1.0), 0.0, math.log(0.5)], dtype=float)
        res = minimize(nll, z0, method="Nelder-Mead", options={"maxiter": 2000})
        kappa, theta, sigma = unpack(res.x)
        return {"kappa": kappa, "theta": theta, "sigma": sigma, "opt_success": bool(res.success)}

    def one_step_metrics(self, p: np.ndarray, dt: float, params: dict) -> tuple[float, float]:
        p = _clip01(p)
        p0, p1 = p[:-1], p[1:]
        kappa, theta, sigma = params["kappa"], params["theta"], params["sigma"]
        m = p0 + kappa * (theta - p0) * dt
        m = _clip01(m)
        v = (sigma**2) * p0 * (1.0 - p0) * dt
        v = np.maximum(v, 1e-12)
        ll = _gaussian_loglik(p1, m, v)
        mse = float(np.mean((p1 - m) ** 2))
        return ll, mse


def _gaussian_loglik(x: np.ndarray, mean: np.ndarray, var: float | np.ndarray) -> float:
    var = np.asarray(var, dtype=float)
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    return float(-0.5 * np.sum(np.log(2.0 * np.pi * var) + ((x - mean) ** 2) / var))


def evaluate_on_market(
    market: dict,
    min_points: int = 1500,
    train_frac: float = 0.8,
    target_dt_s: int = 60,
) -> tuple[str, list[FitResult]] | None:
    outcomes = market.get("outcomes") or []
    token_ids = market.get("clobTokenIds") or []
    # Gamma commonly returns these as JSON-encoded strings.
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            outcomes = []
    if isinstance(token_ids, str):
        try:
            token_ids = json.loads(token_ids)
        except Exception:
            token_ids = []
    if len(outcomes) != 2 or len(token_ids) != 2:
        return None

    # Prefer Yes/No markets for interpretability; otherwise take outcome[0].
    idx = 0
    if outcomes[0].lower() == "no" and outcomes[1].lower() == "yes":
        idx = 1
    token_id = str(token_ids[idx])
    outcome = str(outcomes[idx])

    df = fetch_price_history(token_id, interval="max")
    if df.empty:
        return None

    df, dt_s = regularize_history(df, target_dt_s=target_dt_s)
    if df.empty or len(df) < min_points:
        return None

    p = df["price"].astype(float).to_numpy()
    n = len(p)
    n_train = int(max(10, math.floor(train_frac * n)))
    p_train, p_test = p[:n_train], p[n_train - 1 :]  # include last train point for 1-step

    dt = dt_s  # seconds

    models = [LogitBrownian(), LogitOU(), JacobiEuler()]
    results: list[FitResult] = []
    for model in models:
        params = model.fit(p_train, dt)
        ll, mse = model.one_step_metrics(p_test, dt, params)
        results.append(
            FitResult(
                model=model.name,
                params=params,
                test_loglik=ll / (len(p_test) - 1),  # per-step
                test_mse_p=mse,
                n_train=len(p_train),
                n_test=len(p_test),
            )
        )

    label = f"{market.get('id')} | {market.get('question','')[:60]} | outcome={outcome}"
    return label, results


def main() -> None:
    _ensure_dirs()

    markets = fetch_gamma_markets(limit=200)
    # Sort by volume if present (more liquid -> cleaner diffusion behavior)
    def vol(m: dict) -> float:
        v = m.get("volumeNum")
        try:
            return float(v)
        except Exception:
            return 0.0

    markets = sorted(markets, key=vol, reverse=True)

    all_rows = []
    kept = 0
    for m in markets:
        out = evaluate_on_market(m)
        if out is None:
            continue
        label, results = out
        for r in results:
            all_rows.append(
                {
                    "market_label": label,
                    "model": r.model,
                    "test_loglik_per_step": r.test_loglik,
                    "test_mse_p": r.test_mse_p,
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                }
            )
        kept += 1
        if kept >= 12:  # enough for a stable comparison while staying fast
            break
        time.sleep(0.1)

    if not all_rows:
        raise SystemExit("No markets with sufficient history found; try lowering min_points.")

    res = pd.DataFrame(all_rows)
    res.to_csv("out/model_scores.csv", index=False)

    # Aggregate comparison
    agg = (
        res.groupby("model", as_index=False)
        .agg(
            avg_loglik=("test_loglik_per_step", "mean"),
            med_loglik=("test_loglik_per_step", "median"),
            avg_mse_p=("test_mse_p", "mean"),
            n_markets=("market_label", "nunique"),
        )
        .sort_values("avg_loglik", ascending=False)
    )
    agg.to_csv("out/model_scores_agg.csv", index=False)

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.bar(agg["model"], agg["avg_loglik"])
    plt.title("Avg out-of-sample loglik (per step)")
    plt.xticks(rotation=30, ha="right")
    plt.subplot(1, 2, 2)
    plt.bar(agg["model"], agg["avg_mse_p"])
    plt.title("Avg out-of-sample MSE (price)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("out/model_comparison.png", dpi=160)
    plt.close()

    # Example plot: pick the first market in results and show its series
    ex_market = res["market_label"].iloc[0]
    # Recover token from label is annoying; just re-evaluate that market deterministically by re-running selection.
    # For the example, use the top-volume market that passed evaluation.
    ex = None
    for m in markets:
        out = evaluate_on_market(m)
        if out is None:
            continue
        ex = (m, out)
        break
    if ex is not None:
        m, (label, _) = ex
        outcomes = m["outcomes"]
        token_ids = m["clobTokenIds"]
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        idx = 0
        if outcomes[0].lower() == "no" and outcomes[1].lower() == "yes":
            idx = 1
        token_id = str(token_ids[idx])
        df = fetch_price_history(token_id, interval="max")
        df, dt_s = regularize_history(df, target_dt_s=60)
        df = df.tail(2500)  # keep plot readable
        plt.figure(figsize=(10, 4))
        plt.plot(df["ts"], df["price"], lw=1.0)
        plt.title(f"Example market price history (token {token_id[:8]}…)\n{label}")
        plt.ylabel("Price / implied probability")
        plt.tight_layout()
        plt.savefig("out/example_market_price.png", dpi=160)
        plt.close()

    best = agg.iloc[0].to_dict()
    with open("out/best_model.txt", "w", encoding="utf-8") as f:
        f.write(f"best_model={best['model']}\n")
        f.write(f"avg_loglik_per_step={best['avg_loglik']}\n")
        f.write(f"avg_mse_p={best['avg_mse_p']}\n")
        f.write(f"n_markets={best['n_markets']}\n")


if __name__ == "__main__":
    main()

