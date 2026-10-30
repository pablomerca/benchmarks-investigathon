"""General-capability benchmarks (MMLU, GSM8K): did SFT break the model's knowledge, reasoning or instruct tuning?

These are *not* reward-hacking benchmarks, so they do not go through Rollout/label/rates. The tasks come
from `inspect_evals` (pinned dataset revisions); this module adds
  - `capability_task`  : builds the task with a fixed, shared subset so every model sees the same questions
  - `load_samples`     : .eval logs -> one row per sample (correct, format followed, truncated, tokens, ...)
  - `summarize`        : accuracy + Clopper-Pearson CI + format/truncation rates per (model, benchmark)
  - `compare_to_base`  : paired comparison of each model against a base model, with a regression flag

Instruct-tuning is checked through `format_rate`: both prompts demand a strict final line (`ANSWER: $X`),
so a model whose fine-tuning eroded instruction following shows it there before it shows it in accuracy.
"""

import re
from pathlib import Path

import pandas as pd
from inspect_ai.log import EvalSample, read_eval_log
from inspect_ai.model import ContentReasoning
from scipy.stats import binomtest

from .stats import clopper_pearson

CAPABILITY_BENCHES = ("mmlu", "gsm8k")
# inspect task names (possibly namespaced "inspect_evals/<name>") -> our benchmark label
_TASK_TO_BENCH = {"mmlu_0_shot": "mmlu", "mmlu_5_shot": "mmlu", "gsm8k": "gsm8k"}
SPLIT = "default"  # log dir layout is runs/<model>/<bench>/<split>; these benchmarks have no splits

_ANSWER_LINE = {
    "mmlu": re.compile(r"ANSWER\s*:\s*\(?[A-D]\b", re.IGNORECASE),
    "gsm8k": re.compile(r"ANSWER\s*:\s*\S"),
}


def bench_of(task_name: str) -> str | None:
    """Capability benchmark for an inspect task name, or None if it is not one of ours."""
    return _TASK_TO_BENCH.get(task_name.rsplit("/", 1)[-1])


def capability_task(bench: str, **kw):
    """Inspect task for `bench`. Options (all from configs/runs.yaml `task_kwargs`):

    mmlu : shots (0|5, default 0), cot (default True), subjects (list), plus anything mmlu_*_shot takes
    gsm8k: fewshot (default 8), plus anything gsm8k takes

    MMLU defaults to cot=True: inspect's non-CoT mode caps the answer at a few tokens, which would cut
    off a thinking model before it leaves its reasoning block and score it as wrong. The dataset order
    is fixed (seed 42), so `--limit N` picks the same N questions for every model.
    """
    if bench == "mmlu":
        from inspect_evals.mmlu import mmlu_0_shot, mmlu_5_shot
        shots = kw.pop("shots", 0)
        kw.setdefault("cot", True)
        if shots not in (0, 5):
            raise ValueError(f"mmlu shots must be 0 or 5, got {shots!r}")
        return (mmlu_5_shot if shots == 5 else mmlu_0_shot)(**kw)
    if bench == "gsm8k":
        from inspect_evals.gsm8k import gsm8k
        kw.setdefault("fewshot", 8)
        return gsm8k(**kw)
    raise ValueError(f"unknown capability benchmark {bench!r}")


def _score_value(sample: EvalSample):
    return next(iter(sample.scores.values())).value if sample.scores else None


def _correct(sample: EvalSample) -> bool:
    v = _score_value(sample)
    return v in ("C", 1, 1.0, True)


def _has_reasoning(sample: EvalSample) -> bool:
    for m in sample.messages:
        if m.role == "assistant" and not isinstance(m.content, str) and any(
                isinstance(c, ContentReasoning) for c in m.content):
            return True
    return False


def _truncated(sample: EvalSample) -> bool:
    out = sample.output
    return bool(out and out.choices and out.choices[0].stop_reason == "max_tokens")


def sample_row(sample: EvalSample, model: str, bench: str, path: str = "") -> dict:
    out = sample.output
    completion = out.completion if out and out.choices else ""
    return dict(
        model=model, benchmark=bench, task_id=str(sample.id), epoch=sample.epoch,
        subject=(sample.metadata or {}).get("subject", ""),
        correct=_correct(sample),
        format_ok=bool(_ANSWER_LINE[bench].search(completion)),
        truncated=_truncated(sample),
        has_reasoning=_has_reasoning(sample),
        output_tokens=int(out.usage.output_tokens or 0) if out and out.usage else 0,
        errored=sample.error is not None,
        log_path=path,
    )


def load_samples(logs_dir: str | Path) -> pd.DataFrame:
    """One row per (model, benchmark, task_id, epoch) from every capability .eval log under `logs_dir`."""
    rows = []
    for p in sorted(Path(logs_dir).rglob("*.eval")):
        log = read_eval_log(str(p))
        bench = bench_of(log.eval.task)
        if bench is None:
            continue
        alias = (log.eval.metadata or {}).get("alias") or log.eval.model
        rows += [sample_row(s, alias, bench, str(p)) for s in log.samples or []]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, benchmark): accuracy with 95% CI, and the instruct-following / health signals.

    `format_rate`   share of responses that end with the requested `ANSWER:` line (instruction following)
    `truncated_rate` share that hit max_tokens (runaway reasoning; accuracy is biased down when high)
    `reasoning_rate` share with a reasoning block (distinguishes thinking-on from thinking-off runs)
    `errors`        samples whose run errored; they count as incorrect
    """
    rows = []
    for (model, bench), g in df.groupby(["model", "benchmark"]):
        n, k = len(g), int(g["correct"].sum())
        lo, hi = clopper_pearson(k, n)
        rows.append(dict(
            model=model, benchmark=bench, n=n, k=k, accuracy=k / n, ci_lo=lo, ci_hi=hi,
            format_rate=g["format_ok"].mean(), truncated_rate=g["truncated"].mean(),
            reasoning_rate=g["has_reasoning"].mean(), mean_output_tokens=g["output_tokens"].mean(),
            errors=int(g["errored"].sum())))
    return pd.DataFrame(rows)


def by_subject(df: pd.DataFrame) -> pd.DataFrame:
    """MMLU accuracy per subject and model (wide). Small n per subject: use it to spot where, not whether."""
    m = df[df.benchmark == "mmlu"]
    return m.pivot_table(index="subject", columns="model", values="correct", aggfunc="mean") if len(m) else m


def compare_to_base(df: pd.DataFrame, base: str, models: list[str] | None = None,
                    margin: float = 0.02, format_margin: float = 0.05, alpha: float = 0.05) -> pd.DataFrame:
    """Paired comparison of each model against `base` on the questions both answered.

    Pairing is by (task_id, epoch), so it needs both models run on the same questions (same `--limit`).
    p-value: exact McNemar (two-sided binomial test on the discordant pairs).
    `flag` is REGRESSION when accuracy dropped by more than `margin` AND the drop is significant, and
    FORMAT when the `ANSWER:` rate dropped by more than `format_margin` (instruction following eroded
    even if accuracy held); both can apply. Otherwise OK. Not significant != no effect: read `delta_ci`.
    """
    rows = []
    for bench, g in df.groupby("benchmark"):
        b = g[g.model == base].set_index(["task_id", "epoch"])
        if b.empty:
            continue
        for model in models or sorted(set(g.model) - {base}):
            m = g[g.model == model].set_index(["task_id", "epoch"])
            idx = b.index.intersection(m.index)
            if len(idx) == 0:
                continue
            bc, mc = b.loc[idx, "correct"].astype(bool), m.loc[idx, "correct"].astype(bool)
            only_base, only_model = int((bc & ~mc).sum()), int((~bc & mc).sum())
            disc = only_base + only_model
            p = binomtest(only_base, disc, 0.5).pvalue if disc else 1.0
            delta = float(mc.mean() - bc.mean())
            # Wald interval on the paired difference: var = (b + c - (b - c)^2 / n) / n^2
            n = len(idx)
            se = ((disc - (only_model - only_base) ** 2 / n) / n ** 2) ** 0.5
            fmt_delta = float(m.loc[idx, "format_ok"].mean() - b.loc[idx, "format_ok"].mean())
            flags = []
            if delta < -margin and p < alpha:
                flags.append("REGRESSION")
            if fmt_delta < -format_margin:
                flags.append("FORMAT")
            rows.append(dict(
                benchmark=bench, base=base, model=model, n_paired=n,
                base_acc=float(bc.mean()), model_acc=float(mc.mean()), delta=delta,
                delta_ci_lo=delta - 1.96 * se, delta_ci_hi=delta + 1.96 * se,
                only_base_correct=only_base, only_model_correct=only_model, p_mcnemar=p,
                format_delta=fmt_delta,
                truncated_delta=float(m.loc[idx, "truncated"].mean() - b.loc[idx, "truncated"].mean()),
                flag="+".join(flags) or "OK"))
    return pd.DataFrame(rows)
