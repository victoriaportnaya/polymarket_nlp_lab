import argparse
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _ensure_out() -> None:
    os.makedirs("out", exist_ok=True)
    mpl_cfg = os.path.abspath(".mplconfig")
    os.makedirs(mpl_cfg, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)
    os.environ.setdefault("MPLBACKEND", "Agg")


def _maybe_import_seaborn():
    try:
        import seaborn as sns  # type: ignore

        return sns
    except Exception:
        return None


def load_exorde(path: str, sample_n: int, seed: int) -> pd.DataFrame:
    cols = ["item_raw_content", "item_url", "created_at_ts", "topic_label", "analysis_sentiment"]
    if path.lower().endswith((".csv", ".tsv")):
        sep = "\t" if path.lower().endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep)
        # If topic_label is absent (some Exorde exports), fall back to analysis_classification label if present.
        if "topic_label" not in df.columns and "analysis_classification" in df.columns:
            # analysis_classification looks like "{'label': 'Sports', 'score': 0.7077}"
            s = df["analysis_classification"].astype(str)
            lab = s.str.extract(r"'label'\s*:\s*'([^']+)'", expand=False)
            if lab.isna().mean() > 0.5:
                lab = s.str.extract(r"\"label\"\s*:\s*\"([^\"]+)\"", expand=False)
            df["topic_label"] = lab
    else:
        # path can be a parquet file OR a directory containing partitioned parquet.
        df = pd.read_parquet(path, columns=[c for c in cols if c in ["item_raw_content", "item_url", "created_at_ts", "topic_label"]])

    if "created_at_ts" not in df.columns and "item_created_at" in df.columns:
        df["created_at_ts"] = df["item_created_at"]

    df["created_at_ts"] = pd.to_datetime(df["created_at_ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at_ts", "item_raw_content", "topic_label"])
    if sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=seed)
    df = df.reset_index(drop=True)
    return df


def clean_links_to_domain(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    def repl(m):
        url = m.group(0)
        # minimal parse: strip scheme, keep host
        url2 = re.sub(r"^https?://", "", url)
        url2 = re.sub(r"^www\.", "", url2)
        host = url2.split("/")[0]
        return f"[LINK:{host}]"

    return re.sub(r"https?://\\S+|www\\.\\S+", repl, text)


def basic_text_clean(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    # fast, vectorized cleanup
    s = s.str.replace(r"\\s+", " ", regex=True).str.strip()
    # link normalization (not vectorizable with function well; apply on sample only)
    return s.map(clean_links_to_domain)


def vader_sentiment(texts: pd.Series) -> np.ndarray:
    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    out = np.empty(len(texts), dtype=float)
    for i, t in enumerate(texts.astype(str).tolist()):
        out[i] = sia.polarity_scores(t).get("compound", 0.0)
    return out


def finbert_sentiment(texts: pd.Series, device: str = "cpu", batch_size: int = 16) -> np.ndarray:
    """
    Uses yiyanghkust/finbert-tone (pos/neg/neutral) and returns an expected-value score:
      score = P(pos) - P(neg)
    Requires network the first time to download the model weights.
    """
    from transformers import pipeline

    pipe = pipeline(
        "text-classification",
        model="yiyanghkust/finbert-tone",
        device=0 if device == "cuda" else -1,
        top_k=None,
        truncation=True,
        max_length=512,
    )
    scores = np.zeros(len(texts), dtype=float)
    docs = texts.astype(str).tolist()
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        res = pipe(batch)
        # res: list[list[{'label':..., 'score':...}, ...]]
        for j, rr in enumerate(res):
            p = {x["label"].lower(): float(x["score"]) for x in rr}
            scores[start + j] = p.get("positive", 0.0) - p.get("negative", 0.0)
    return scores


@dataclass
class EDAConfig:
    top_k_topics: int
    min_text_len: int
    resample_rule: str


def make_plots(df: pd.DataFrame, out_prefix: str, cfg: EDAConfig) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sns = _maybe_import_seaborn()
    if sns is not None:
        sns.set_theme()

    # Top topics
    top = df.groupby("topic_label")["item_raw_content"].count().sort_values(ascending=False).head(cfg.top_k_topics)
    plt.figure(figsize=(10, 6))
    if sns is not None:
        sns.barplot(x=top.values, y=top.index)
    else:
        plt.barh(top.index[::-1], top.values[::-1])
    plt.title(f"Top {cfg.top_k_topics} Topics (sample)")
    plt.tight_layout()
    plt.savefig(f"out/{out_prefix}_top_topics.png", dpi=160)
    plt.close()

    # Hourly volume
    hourly = df.set_index("created_at_ts").resample(cfg.resample_rule)["item_raw_content"].count().rename("message_count")
    plt.figure(figsize=(12, 4))
    plt.plot(hourly.index, hourly.values, lw=1.0)
    plt.title(f"Message volume ({cfg.resample_rule})")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"out/{out_prefix}_volume.png", dpi=160)
    plt.close()

    # Sentiment distribution
    if "sentiment" in df.columns:
        plt.figure(figsize=(8, 3.5))
        if sns is not None:
            sns.kdeplot(df["sentiment"], fill=True)
        else:
            plt.hist(df["sentiment"], bins=60, density=True, alpha=0.8)
        plt.title("Sentiment distribution")
        plt.tight_layout()
        plt.savefig(f"out/{out_prefix}_sentiment_dist.png", dpi=160)
        plt.close()

        # Sentiment by topic (top_k)
        top_topics = top.index.tolist()
        grouped = (
            df[df["topic_label"].isin(top_topics)]
            .groupby("topic_label")["sentiment"]
            .mean()
            .sort_values()
            .reset_index()
        )
        plt.figure(figsize=(10, 6))
        if sns is not None:
            sns.barplot(data=grouped, x="sentiment", y="topic_label")
        else:
            plt.barh(grouped["topic_label"], grouped["sentiment"])
        plt.axvline(0, color="k", lw=1)
        plt.title("Mean sentiment by topic (top topics)")
        plt.tight_layout()
        plt.savefig(f"out/{out_prefix}_sentiment_by_topic.png", dpi=160)
        plt.close()


def main() -> None:
    _ensure_out()
    ap = argparse.ArgumentParser()
    ap.add_argument("--exorde_parquet", type=str, required=True, help="Path to Exorde parquet file or directory")
    ap.add_argument("--sample_n", type=int, default=200_000, help="Row sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min_text_len", type=int, default=20)
    ap.add_argument("--sentiment_model", type=str, default="vader", help="vader|finbert")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda (for finbert)")
    ap.add_argument("--batch_size", type=int, default=16, help="finbert batch size")
    ap.add_argument("--resample", type=str, default="1H", help="Pandas resample rule, e.g. 1H")
    ap.add_argument("--out_prefix", type=str, default="exorde", help="Prefix for out files")
    ap.add_argument(
        "--use_precomputed_sentiment",
        action="store_true",
        help="If Exorde CSV contains analysis_sentiment, use it instead of recomputing sentiment from text.",
    )
    ap.add_argument("--topic_include_regex", type=str, default="", help="Keep only rows whose topic_label matches this regex")
    ap.add_argument("--domain_include_regex", type=str, default="", help="Keep only rows whose domain_bucket/domain matches this regex")
    args = ap.parse_args()
    # Newer pandas versions prefer lowercase frequency aliases (e.g. '1h' not '1H').
    args.resample = args.resample.replace("H", "h").replace("D", "d").replace("T", "min")

    df = load_exorde(args.exorde_parquet, sample_n=args.sample_n, seed=args.seed)
    if args.topic_include_regex:
        df = df[df["topic_label"].astype(str).str.contains(args.topic_include_regex, regex=True, na=False)].copy()
    if args.domain_include_regex:
        dom_col = "domain_bucket" if "domain_bucket" in df.columns else ("domain" if "domain" in df.columns else "")
        if dom_col:
            df = df[df[dom_col].astype(str).str.contains(args.domain_include_regex, regex=True, na=False)].copy()
    df["item_raw_content"] = df["item_raw_content"].astype(str)
    df = df[df["item_raw_content"].str.len() >= args.min_text_len].copy()

    # Minimal cleaning (KISS, but similar spirit to the notebook): normalize whitespace + link domains.
    df["content_clean"] = basic_text_clean(df["item_raw_content"])

    # Sentiment
    if args.use_precomputed_sentiment and "analysis_sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["analysis_sentiment"], errors="coerce").fillna(0.0).astype(float)
    else:
        if args.sentiment_model == "finbert":
            df["sentiment"] = finbert_sentiment(df["content_clean"], device=args.device, batch_size=args.batch_size)
        elif args.sentiment_model == "vader":
            df["sentiment"] = vader_sentiment(df["content_clean"])
        else:
            raise SystemExit("--sentiment_model must be vader or finbert")

    # EDA plots
    make_plots(
        df,
        out_prefix=args.out_prefix,
        cfg=EDAConfig(top_k_topics=30, min_text_len=args.min_text_len, resample_rule=args.resample),
    )

    # Aggregate to hourly sentiment series suitable for our signature scripts: columns ts,sentiment
    hourly = (
        df.set_index("created_at_ts")
        .resample(args.resample)
        .agg(sentiment=("sentiment", "mean"), message_count=("sentiment", "size"))
        .dropna()
        .reset_index()
        .rename(columns={"created_at_ts": "ts"})
    )
    hourly.to_csv(f"out/{args.out_prefix}_sentiment_{args.resample}.csv", index=False)
    try:
        hourly.to_parquet(f"out/{args.out_prefix}_sentiment_{args.resample}.parquet", index=False)
    except ImportError:
        # Parquet engine (pyarrow/fastparquet) not available in this environment; CSV is enough for downstream.
        pass


if __name__ == "__main__":
    main()

