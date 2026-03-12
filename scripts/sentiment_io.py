import pandas as pd


def load_sentiment_series(
    sentiment_csv: str,
    ts_col: str = "ts",
    value_col: str = "sentiment",
    tz: str | None = "UTC",
) -> pd.Series:
    """
    Load a sentiment time series from CSV.

    Expected columns by default:
    - ts: timestamp (ISO8601 or any pandas-parseable datetime)
    - sentiment: numeric sentiment score (can be any scale; we'll standardize later)
    """
    df = pd.read_csv(sentiment_csv)
    if ts_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"CSV must contain columns {ts_col!r} and {value_col!r}")

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if tz is not None and tz.upper() != "UTC":
        ts = ts.dt.tz_convert(tz)
    s = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.Series(s.values, index=ts).dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def align_sentiment_to_index(
    sentiment: pd.Series,
    target_index: pd.DatetimeIndex,
    method: str = "ffill",
    fill_value: float = 0.0,
) -> pd.Series:
    """
    Align sentiment values onto a target timestamp index.
    - method=ffill: use last-known sentiment value
    - method=nearest: nearest neighbor
    - method=zero: fill missing with fill_value
    """
    s = sentiment.sort_index()
    if method == "nearest":
        aligned = s.reindex(target_index, method="nearest", tolerance=None)
    elif method == "ffill":
        aligned = s.reindex(target_index, method="ffill")
    elif method == "zero":
        aligned = s.reindex(target_index)
    else:
        raise ValueError("method must be one of: ffill, nearest, zero")

    return aligned.fillna(fill_value)

