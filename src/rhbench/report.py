"""CLI: logs dir -> rollouts.parquet + rates.csv (+ printed table)."""

import argparse
from pathlib import Path

import pandas as pd

from .extract import load_rollouts
from .stats import rates


def build(logs_dir: str, out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = load_rollouts(logs_dir)
    if not rows:
        raise SystemExit(f"no .eval logs found under {logs_dir}")
    df = pd.DataFrame([r.to_row() for r in rows])
    rt = rates(df)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "rollouts.parquet", index=False)
    rt.to_csv(out / "rates.csv", index=False)
    return df, rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="runs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df, rt = build(a.logs, a.out)
    print(f"{len(df)} rollouts -> {a.out}/rollouts.parquet")
    print(rt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
