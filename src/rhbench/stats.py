"""Rates with Clopper-Pearson CIs and Fisher exact comparison (RHB paper, sec. 5.4)."""

import pandas as pd
from scipy.stats import beta, fisher_exact

from .schema import HACKED, IMPOSSIBLE_SPLITS, SOLVED


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def rates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, benchmark, split, agent_type).

    `rate` is the hack rate on impossible splits and the solve rate on `original`
    (kept apart by the `metric` column so they are never mixed).
    """
    rows = []
    for (model, bench, split, agent), g in df.groupby(["model", "benchmark", "split", "agent_type"]):
        target = HACKED if split in IMPOSSIBLE_SPLITS else SOLVED
        n, k = len(g), int((g["label"] == target).sum())
        lo, hi = clopper_pearson(k, n)
        rows.append(dict(model=model, benchmark=bench, split=split, agent_type=agent,
                         metric="hack_rate" if target == HACKED else "solve_rate",
                         n=n, k=k, rate=k / n, ci_lo=lo, ci_hi=hi,
                         unsolved=int((g["label"] == "Unsolved").sum()),
                         test_file_modified=int(g["test_file_modified"].sum())))
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, model_a: str, model_b: str, benchmark: str, split: str) -> float:
    """Two-sided Fisher exact p-value for target-label frequency, model_a vs model_b."""
    target = HACKED if split in IMPOSSIBLE_SPLITS else SOLVED
    tab = []
    for m in (model_a, model_b):
        g = df[(df.model == m) & (df.benchmark == benchmark) & (df.split == split)]
        k = int((g.label == target).sum())
        tab.append([k, len(g) - k])
    return float(fisher_exact(tab)[1])
