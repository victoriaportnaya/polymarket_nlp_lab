import argparse
import datetime as dt
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests


def fetch_pageviews(
    article: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    project: str = "en.wikipedia.org",
    access: str = "all-access",
    agent: str = "user",
    granularity: str = "hourly",
    timeout_s: int = 30,
    verify_ssl: bool = False,
    min_wait_s: float = 0.25,
) -> pd.DataFrame:
    """
    Uses Wikimedia Pageviews API:
      https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}
    """
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    art = quote(article, safe="")
    start_s = start.strftime("%Y%m%d%H")
    end_s = end.strftime("%Y%m%d%H")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{project}/{access}/{agent}/{art}/{granularity}/{start_s}/{end_s}"
    )
    headers = {"User-Agent": "polymarket-research/1.0 (academic)"}  # required by Wikimedia policy

    time.sleep(min_wait_s)
    r = requests.get(url, headers=headers, timeout=timeout_s, verify=verify_ssl)
    r.raise_for_status()
    j = r.json()
    items = j.get("items", [])
    rows = []
    for it in items:
        try:
            ts = dt.datetime.strptime(it["timestamp"], "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
            rows.append((ts, int(it.get("views", 0))))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["ts", article]).sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, required=True, help="ISO (UTC), e.g. 2026-02-07T20:00:00Z")
    ap.add_argument("--end", type=str, required=True, help="ISO (UTC), e.g. 2026-03-07T19:59:00Z")
    ap.add_argument(
        "--articles",
        type=str,
        default="Ukraine,Russia,Russo-Ukrainian_War,Volodymyr_Zelenskyy,Vladimir_Putin,Ceasefire",
        help="Comma-separated Wikipedia article titles (use underscores).",
    )
    ap.add_argument("--out_csv", type=str, default="out/wiki_ukraine_intensity_1h.csv")
    ap.add_argument("--resample", type=str, default="1h")
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    end = dt.datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    if end <= start:
        raise SystemExit("end must be > start")

    articles = [a.strip() for a in args.articles.split(",") if a.strip()]
    if not articles:
        raise SystemExit("No articles provided.")

    dfs = []
    for a in articles:
        df = fetch_pageviews(a, start, end)
        if not df.empty:
            dfs.append(df.set_index("ts"))

    if not dfs:
        raise SystemExit("No pageview data returned.")

    wide = pd.concat(dfs, axis=1).fillna(0.0)
    wide = wide.resample(args.resample).sum()

    # intensity proxy: log(1 + total_views), standardized
    total = wide.sum(axis=1).astype(float)
    proxy = np.log1p(total.to_numpy())
    proxy = (proxy - float(np.mean(proxy))) / (float(np.std(proxy)) + 1e-12)

    out = pd.DataFrame({"ts": wide.index, "sentiment": proxy})
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} rows -> {args.out_csv}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

