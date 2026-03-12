import argparse
import os
import subprocess


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exorde_parquet", type=str, required=True)
    ap.add_argument("--sample_n", type=int, default=200_000)
    ap.add_argument("--sentiment_model", type=str, default="vader", help="vader|finbert")
    ap.add_argument("--use_precomputed_sentiment", action="store_true", help="Use Exorde analysis_sentiment if present")
    ap.add_argument("--resample", type=str, default="1h")
    ap.add_argument("--out_prefix", type=str, default="exorde")
    ap.add_argument("--market_limit", type=int, default=200, help="Gamma markets to scan")
    ap.add_argument("--topic_include_regex", type=str, default="", help="Topic regex filter forwarded to exorde_sentiment_eda.py")
    args = ap.parse_args()

    out_csv = os.path.abspath(f"out/{args.out_prefix}_sentiment_{args.resample}.csv")

    cmd = [
        "python3",
        "exorde_sentiment_eda.py",
        "--exorde_parquet",
        args.exorde_parquet,
        "--sample_n",
        str(args.sample_n),
        "--sentiment_model",
        args.sentiment_model,
        "--resample",
        args.resample,
        "--out_prefix",
        args.out_prefix,
    ]
    if args.use_precomputed_sentiment:
        cmd.append("--use_precomputed_sentiment")
    if args.topic_include_regex:
        cmd += ["--topic_include_regex", args.topic_include_regex]
    run(cmd)

    # Lead–lag signature sweep (uses real sentiment csv)
    run(
        [
            "python3",
            "polymarket_signature_experiments.py",
            "--limit",
            str(args.market_limit),
            "--sentiment_csv",
            out_csv,
            "--sentiment_ts_col",
            "ts",
            "--sentiment_value_col",
            "sentiment",
            "--sentiment_align",
            "ffill",
        ]
    )

    # Teichmann randomized signature reconstruction (uses real sentiment csv)
    run(
        [
            "python3",
            "polymarket_randomized_signature_reconstruction.py",
            "--limit",
            str(args.market_limit),
            "--sentiment_csv",
            out_csv,
            "--sentiment_ts_col",
            "ts",
            "--sentiment_value_col",
            "sentiment",
            "--sentiment_align",
            "ffill",
        ]
    )


if __name__ == "__main__":
    main()

