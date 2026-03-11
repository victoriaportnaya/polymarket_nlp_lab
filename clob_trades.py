import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from clob_l2_auth import PolyL2Creds, make_l2_headers


END_CURSOR = "LTE="


@dataclass(frozen=True)
class TradeQuery:
    market: str | None = None  # token id
    asset_id: str | None = None
    maker_address: str | None = None
    trade_id: str | None = None
    after: int | None = None
    before: int | None = None


def _build_query_params(base_url: str, q: TradeQuery, next_cursor: str) -> str:
    # mirrors py-clob-client add_query_trade_params
    url = base_url
    has_query = bool(next_cursor) or any(
        [q.market, q.asset_id, q.after, q.before, q.maker_address, q.trade_id]
    )
    if has_query:
        url += "?"

    def add(param: str, val: Any) -> None:
        nonlocal url
        if url.endswith("?"):
            url += f"{param}={val}"
        else:
            url += f"&{param}={val}"

    if q.market:
        add("market", q.market)
    if q.asset_id:
        add("asset_id", q.asset_id)
    if q.after:
        add("after", q.after)
    if q.before:
        add("before", q.before)
    if q.maker_address:
        add("maker_address", q.maker_address)
    if q.trade_id:
        add("id", q.trade_id)
    if next_cursor:
        add("next_cursor", next_cursor)
    return url


def fetch_market_trades(
    creds: PolyL2Creds,
    market_token_id: str,
    after_ts: int,
    before_ts: int,
    timeout_s: int = 30,
) -> list[dict[str, Any]]:
    """
    Fetches trades from /data/trades for a specific market token id using L2 auth.
    """
    base = "https://clob.polymarket.com/data/trades"
    request_path = "/data/trades"

    q = TradeQuery(market=str(market_token_id), after=int(after_ts), before=int(before_ts))
    next_cursor = "MA=="
    out: list[dict[str, Any]] = []

    # sign only the path (no query), per py-clob-client
    headers = make_l2_headers(creds, method="GET", request_path=request_path)
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    while next_cursor != END_CURSOR:
        url = _build_query_params(base, q, next_cursor=next_cursor)
        r = requests.get(url, headers=headers, timeout=timeout_s, verify=False)
        r.raise_for_status()
        j = r.json()
        next_cursor = j.get("next_cursor", END_CURSOR)
        data = j.get("data", [])
        if not isinstance(data, list):
            break
        out.extend([x for x in data if isinstance(x, dict)])
        if not data and next_cursor == END_CURSOR:
            break

    return out


def trades_to_price_df(trades: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Best-effort conversion of CLOB trade objects into a time series of last-trade prices.
    We intentionally keep this permissive and inspect common keys.
    """
    if not trades:
        return pd.DataFrame(columns=["ts", "price"])

    # common candidates across APIs
    t_keys = ["timestamp", "ts", "t", "created_at", "createdAt", "time"]
    p_keys = ["price", "p"]

    rows = []
    for tr in trades:
        t_val = None
        for k in t_keys:
            if k in tr:
                t_val = tr.get(k)
                break
        p_val = None
        for k in p_keys:
            if k in tr:
                p_val = tr.get(k)
                break

        if t_val is None or p_val is None:
            continue

        try:
            # many APIs return seconds; if ms, convert
            t = int(float(t_val))
            if t > 10_000_000_000:  # ms
                t = t // 1000
            ts = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
            price = float(p_val)
        except Exception:
            continue
        rows.append((ts, price))

    if not rows:
        return pd.DataFrame(columns=["ts", "price"])

    df = pd.DataFrame(rows, columns=["ts", "price"]).sort_values("ts").drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.reset_index(drop=True)

