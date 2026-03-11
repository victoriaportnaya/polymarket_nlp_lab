import argparse
import datetime as dt
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


def _gdelt_doc_timeline(
    query: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    mode: str = "timelinevolraw",
    verify_ssl: bool = False,
    timeout_s: int = 30,
    min_wait_s: float = 6.0,
    max_retries: int = 6,
    throttle_sleep_s: float = 70.0,
) -> dict[str, Any]:
    """
    Calls GDELT DOC 2.1 timeline endpoint. Obeys simple rate-limiting (>=5s).
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    headers = {"User-Agent": "polymarket-research/1.0 (offline academic)"}
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    # GDELT explicitly asks for <= 1 request / 5 seconds. We'll be conservative and,
    # on 429, pause for a longer cooldown.
    for attempt in range(max_retries):
        time.sleep(min_wait_s)
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout_s, verify=verify_ssl)
        except requests.RequestException as e:
            # transient DNS/network hiccup
            print(f"[gdelt] request error attempt={attempt+1}/{max_retries}: {e!r} (sleep {throttle_sleep_s}s)")
            time.sleep(throttle_sleep_s)
            continue

        if r.status_code == 200:
            return r.json()

        body = (r.text or "")[:200].replace("\n", " ")
        if r.status_code == 429:
            print(f"[gdelt] throttled (429) attempt={attempt+1}/{max_retries} cooldown={throttle_sleep_s}s body='{body}'")
            time.sleep(throttle_sleep_s)
            continue

        # Other transient errors
        print(f"[gdelt] status={r.status_code} attempt={attempt+1}/{max_retries} cooldown={throttle_sleep_s}s body='{body}'")
        time.sleep(throttle_sleep_s)

    raise RuntimeError("GDELT request failed after retries.")


def timeline_to_df(j: dict[str, Any]) -> pd.DataFrame:
    # Expect j["timeline"] -> [{series:..., data:[{date,value,norm}, ...]}]
    tl = j.get("timeline", [])
    if not isinstance(tl, list) or not tl:
        return pd.DataFrame(columns=["ts", "value", "norm"])
    data = tl[0].get("data", [])
    if not isinstance(data, list):
        return pd.DataFrame(columns=["ts", "value", "norm"])

    rows = []
    for d in data:
        try:
            ts = dt.datetime.strptime(d["date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
            rows.append((ts, float(d.get("value", 0.0)), float(d.get("norm", 0.0))))
        except Exception:
            continue
    return pd.DataFrame(rows, columns=["ts", "value", "norm"]).sort_values("ts").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--start", type=str, required=True, help="ISO (UTC), e.g. 2026-02-07T00:00:00Z")
    ap.add_argument("--end", type=str, required=True, help="ISO (UTC), e.g. 2026-03-07T00:00:00Z")
    ap.add_argument("--out_csv", type=str, default="out/gdelt_sentiment_1h.csv")
    ap.add_argument("--resample", type=str, default="1h", help="pandas resample string, e.g. 1h")
    ap.add_argument(
        "--chunk_hours",
        type=int,
        default=24,
        help="Chunk the GDELT request into this many hours per call to avoid server-side downsampling.",
    )
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    end = dt.datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    if end <= start:
        raise SystemExit("end must be > start")

    # Chunk to avoid GDELT returning only daily resolution on long windows.
    chunks: list[pd.DataFrame] = []
    cur = start
    delta = dt.timedelta(hours=int(args.chunk_hours))
    while cur < end:
        nxt = min(end, cur + delta)
        j = _gdelt_doc_timeline(args.query, cur, nxt)
        dfi = timeline_to_df(j)
        if not dfi.empty:
            chunks.append(dfi)
        cur = nxt

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["ts", "value", "norm"])
    if df.empty:
        raise SystemExit("No timeline data returned.")
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    # Use normalized attention proxy (value / norm), then log1p and z-score.
    denom = df["norm"].replace(0.0, np.nan)
    attn = (df["value"] / denom).fillna(0.0)
    proxy = np.log1p(attn.to_numpy())

    dfp = df[["ts"]].copy()
    dfp["proxy"] = proxy
    dfp["ts"] = pd.to_datetime(dfp["ts"], utc=True)
    dfp = dfp.set_index("ts").resample(args.resample).mean().dropna()

    s = dfp["proxy"].to_numpy()
    s = (s - float(np.mean(s))) / (float(np.std(s)) + 1e-12)
    out = dfp.reset_index().rename(columns={"proxy": "sentiment"})[["ts", "sentiment"]]
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} rows -> {args.out_csv}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

