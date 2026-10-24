"""Rates with Clopper-Pearson CIs and Fisher exact comparison (RHB paper, sec. 5.4)."""

import pandas as pd
from scipy.stats import beta, fisher_exact

from .schema import HACKED, IMPOSSIBLE_SPLITS, NATURAL, SOLVED


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def _row(model, bench, split, agent, metric, n, k, g):
    lo, hi = clopper_pearson(k, n)
    return dict(model=model, benchmark=bench, split=split, agent_type=agent, metric=metric,
                n=n, k=k, rate=k / n if n else float("nan"), ci_lo=lo, ci_hi=hi,
                unsolved=int((g["label"] == "Unsolved").sum()),
                hack_signal=int(g["hack_signal"].sum()),
                test_file_modified=int(g["test_file_modified"].sum()),
                errors=int((g["exit_status"] == "error").sum()))


def trajectory_table(g: pd.DataFrame) -> pd.DataFrame:
    """HVTA: one row per trajectory; hacked if ANY of its games fired a hack (the paper's HR unit)."""
    return g.groupby("trajectory_id", as_index=False).agg(hacked=("hack_signal", "any"))


def _hvta_rows(model, bench, split, agent, g):
    traj = trajectory_table(g)
    clean = g[~g["hack_signal"].astype(bool)]
    return [
        # trajectory level (n = trajectories): the paper's Hack Rate
        _row(model, bench, split, agent, "hack_rate", len(traj), int(traj["hacked"].sum()), g),
        # game level over games where no hack fired (n = clean games): the paper's Hack-Free Win Rate.
        # Pooled over game indices; the paper averages the per-index conditional rates instead.
        _row(model, bench, split, agent, "hack_free_win_rate", len(clean), int(clean["won"].sum()), g),
    ]


def rates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, benchmark, split, agent_type, metric).

    Impossible splits -> hack_rate; `original` -> solve_rate. EvilGenie (`natural`) emits both
    hack_rate and solve_rate from the same rollouts, plus `hack_signal` (attempts, incl. failed).
    HVTA emits trajectory-level hack_rate and game-level hack_free_win_rate (see _hvta_rows).
    They are kept in separate rows and never mixed. `errors` counts rollouts whose run errored: those
    are labelled Unsolved, so a non-zero value means the rates are biased downwards.
    """
    rows = []
    for (model, bench, split, agent), g in df.groupby(["model", "benchmark", "split", "agent_type"]):
        if bench == "hvta":
            rows += _hvta_rows(model, bench, split, agent, g)
            continue
        targets = ([("hack_rate", HACKED), ("solve_rate", SOLVED)] if split == NATURAL
                   else [("hack_rate", HACKED)] if split in IMPOSSIBLE_SPLITS else [("solve_rate", SOLVED)])
        for metric, target in targets:
            rows.append(_row(model, bench, split, agent, metric, len(g), int((g["label"] == target).sum()), g))
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, model_a: str, model_b: str, benchmark: str, split: str) -> float:
    """Two-sided Fisher exact p-value for target-label frequency, model_a vs model_b.

    Treats units as independent. For HVTA the unit is the trajectory, but trajectories of the same
    environment are correlated, so prefer a bootstrap over environments for real conclusions.
    """
    target = HACKED if split in IMPOSSIBLE_SPLITS or split == NATURAL else SOLVED
    tab = []
    for m in (model_a, model_b):
        g = df[(df.model == m) & (df.benchmark == benchmark) & (df.split == split)]
        if benchmark == "hvta":
            t = trajectory_table(g)
            k, n = int(t["hacked"].sum()), len(t)
        else:
            k, n = int((g.label == target).sum()), len(g)
        tab.append([k, n - k])
    return float(fisher_exact(tab)[1])
