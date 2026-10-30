"""CLI: capability logs (MMLU, GSM8K) -> results/capability*.csv + printed base-vs-SFT comparison.

  python -m rhbench.capability_report --logs runs --base qwen-base
  python -m rhbench.capability_report --logs runs --base qwen-base --models qwen-sorh2 --out results
"""

import argparse
from pathlib import Path

import pandas as pd

from .capability import by_subject, compare_to_base, load_samples, summarize


def build(logs_dir: str, out_dir: str, base: str | None = None, models: list[str] | None = None,
          margin: float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    df = load_samples(logs_dir)
    if df.empty:
        raise SystemExit(f"no MMLU/GSM8K .eval logs found under {logs_dir}")
    if base and base not in set(df.model):
        raise SystemExit(f"base model {base!r} has no capability logs; found: {sorted(set(df.model))}")
    summary = summarize(df)
    cmp = compare_to_base(df, base, models, margin=margin) if base else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "capability_samples.parquet", index=False)
    summary.to_csv(out / "capability.csv", index=False)
    by_subject(df).to_csv(out / "capability_mmlu_subjects.csv")
    if cmp is not None:
        cmp.to_csv(out / "capability_compare.csv", index=False)
    return df, summary, cmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--base", default=None, help="model alias to compare everything else against")
    ap.add_argument("--models", nargs="+", default=None, help="only compare these (default: all but base)")
    ap.add_argument("--margin", type=float, default=0.02, help="accuracy drop that counts as a regression")
    a = ap.parse_args()
    df, summary, cmp = build(a.logs, a.out, a.base, a.models, a.margin)
    print(f"{len(df)} samples -> {a.out}/capability_samples.parquet\n")
    print(summary.round(3).to_string(index=False))
    if cmp is not None:
        print("\nvs base:\n" + (cmp.round(3).to_string(index=False) if len(cmp)
                                else "(no model shares questions with the base: run with the same --limit)"))


if __name__ == "__main__":
    main()
