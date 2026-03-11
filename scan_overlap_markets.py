import argparse
import csv
import datetime as dt
import json
import re
from typing import Any

import requests


def _json_get(url: str, params: dict[str, str], timeout_s: int = 30) -> Any:
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    r = requests.get(url, params=params, timeout=timeout_s, verify=False)
    r.raise_for_status()
    return r.json()


def _parse_list_field(x: Any) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def _compile_keyword_pattern(keywords: list[str]) -> re.Pattern:
    """
    Build a keyword regex that avoids substring false positives.
    - alphabetic keywords are matched on word boundaries (e.g. "eth" won't match "Beth")
    - non-alphabetic tokens use literal matching
    """
    parts: list[str] = []
    for k in keywords:
        token = k.strip()
        if not token:
            continue
        esc = re.escape(token)
        if token.isalpha():
            parts.append(rf"\b{esc}\b")
        else:
            parts.append(esc)
    if not parts:
        # never-match fallback
        return re.compile(r"a^")
    return re.compile("|".join(parts), flags=re.IGNORECASE)


def fetch_price_points_in_window(token_id: str, start_ts: int, end_ts: int) -> tuple[int, int, int]:
    """
    Return (n_in_window, min_t, max_t) after filtering returned points to [start_ts, end_ts].
    Use interval=max because other intervals appear to ignore the window in practice.
    """
    j = _json_get(
        "https://clob.polymarket.com/prices-history",
        params={"market": token_id, "interval": "max", "startTs": str(start_ts), "endTs": str(end_ts)},
    )
    hist = j.get("history", [])
    in_w = [h for h in hist if start_ts <= int(h.get("t", -1)) <= end_ts]
    if not in_w:
        return 0, 0, 0
    ts = [int(h["t"]) for h in in_w]
    return len(in_w), min(ts), max(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, required=True, help="Start time ISO, e.g. 2025-12-01T00:00:00Z")
    ap.add_argument("--end", type=str, required=True, help="End time ISO, e.g. 2025-12-02T00:00:00Z")
    ap.add_argument(
        "--keywords",
        type=str,
        default="ukraine,russia,russian,kyiv,kiev,zelensky,putin,kremlin,donbas,crimea,ceasefire,truce,nato",
    )
    ap.add_argument("--max_offsets", type=int, default=40, help="How many pages to scan (each page is limit)")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--include_closed", action="store_true", help="Also scan closed=true markets")
    ap.add_argument("--out_csv", type=str, default="out/overlap_scan.csv")
    args = ap.parse_args()

    start_dt = dt.datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end_dt = dt.datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    kw = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    pat = _compile_keyword_pattern(kw)

    rows = []
    modes = [("false", "open")]
    if args.include_closed:
        modes.append(("true", "closed"))

    for closed_val, mode in modes:
        for page in range(args.max_offsets):
            offset = page * args.limit
            markets = _json_get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": closed_val, "limit": str(args.limit), "offset": str(offset)},
            )
            if not isinstance(markets, list) or not markets:
                break

            for m in markets:
                q = (m.get("question") or "")
                if not pat.search(q):
                    continue
                outcomes = _parse_list_field(m.get("outcomes"))
                tids = _parse_list_field(m.get("clobTokenIds"))
                if len(outcomes) != 2 or len(tids) != 2:
                    continue

                # prefer YES token if present
                idx = 0
                if outcomes[0].lower() == "no" and outcomes[1].lower() == "yes":
                    idx = 1
                elif outcomes[0].lower() == "yes":
                    idx = 0
                elif outcomes[1].lower() == "yes":
                    idx = 1

                token_id = str(tids[idx])
                try:
                    n_in, tmin, tmax = fetch_price_points_in_window(token_id, start_ts, end_ts)
                except Exception:
                    n_in, tmin, tmax = 0, 0, 0

                rows.append(
                    {
                        "mode": mode,
                        "market_id": str(m.get("id")),
                        "question": q,
                        "token_id": token_id,
                        "outcome": str(outcomes[idx]),
                        "n_points_in_window": int(n_in),
                        "min_t": int(tmin),
                        "max_t": int(tmax),
                    }
                )

    rows = sorted(rows, key=lambda r: r["n_points_in_window"], reverse=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["mode"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    top = rows[:10]
    print("Top candidates by n_points_in_window:")
    for r in top:
        print(r["n_points_in_window"], r["market_id"], r["question"][:80])


if __name__ == "__main__":
    main()

