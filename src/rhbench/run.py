"""CLI: run a benchmark for a model over a (benchmark x split) grid.

  python -m rhbench.run --model mock:cheater --bench lcb --split oneoff conflicting --limit 5
  python -m rhbench.run --model qwen-small --bench lcb --split original oneoff conflicting
  python -m rhbench.run --model qwen-small --bench eg --judge claude-judge --limit 20
  python -m rhbench.run --model qwen-small --bench eg --no-judge        # tests + file integrity only
  python -m rhbench.run --model qwen-small --bench hvta --limit 5        # text games, exact hack flags
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from inspect_ai import eval

from .models import CONFIGS, is_mock, resolve, resolve_judge
from .schema import NATURAL, SPLITS

# EvilGenie is third-party code, cloned (not vendored) next to this package; see README.
EVILGENIE_DIR = Path(os.environ.get(
    "EVILGENIE_DIR", Path(__file__).resolve().parents[2] / "external" / "evilgenie_inspect"))


def _evilgenie_task(limit: int | None, **kw):
    if not (EVILGENIE_DIR / "reward_hacking.py").exists():
        raise FileNotFoundError(
            f"EvilGenie not found at {EVILGENIE_DIR}. Clone it:\n"
            "  git clone https://github.com/JonathanGabor/evilgenie_inspect external/evilgenie_inspect")
    # Its modules import each other by bare name (constants, dataset, tools, ...). Append, so
    # nothing in site-packages is shadowed.
    if str(EVILGENIE_DIR) not in sys.path:
        sys.path.append(str(EVILGENIE_DIR))
    from reward_hacking import reward_hacking
    return reward_hacking(limit=limit, **kw)


# Benchmarks without ImpossibleBench-style splits: run once, with a fixed agent-type label.
# (HVTA's two hack types live inside the one task and become the `split` column per sample.)
SPLITLESS = {"eg": ("basic", NATURAL), "hvta": ("text", NATURAL)}


def make_task(bench: str, split: str, agent_type: str, limit: int | None, **kw):
    if bench == "eg":
        return _evilgenie_task(limit, **kw)
    if bench == "hvta":
        from .hvta_task import hvta
        return hvta(limit=limit, **kw)
    from impossiblebench import impossible_livecodebench, impossible_swebench
    if bench == "lcb":
        return impossible_livecodebench(split, agent_type=agent_type, limit=limit, **kw)
    if bench == "swe":
        return impossible_swebench(split, agent_type=agent_type, limit=limit, **kw)
    raise ValueError(f"unknown benchmark {bench!r}")


def run_grid(model: str, benches, splits, agent_type="minimal", limit=None, epochs=1,
             log_root="runs", task_kwargs=None, max_samples=None, judge: str | None = None,
             no_judge: bool = False):
    """Returns the list of EvalLogs. One log dir per (model, bench, split).

    `splits` only applies to ImpossibleBench; EvilGenie ("eg") has none and always runs once.
    EvilGenie needs a judge model (`judge`, an alias in models.yaml or `mock-judge:<verdict>`)
    unless `no_judge` is set; there is deliberately no silent fallback to its OpenAI default.
    """
    logs = []
    for bench in benches:
        is_eg = bench == "eg"
        splitless = bench in SPLITLESS
        if is_eg and not judge and not no_judge:
            raise ValueError("EvilGenie needs --judge <alias> (e.g. a Claude entry in models.yaml) "
                             "or --no-judge to skip the LLM judge")
        for split in ([SPLITLESS[bench][1]] if splitless else splits):
            agent = SPLITLESS[bench][0] if splitless else agent_type
            kw = dict((task_kwargs or {}).get(bench, {}))
            if is_eg:
                kw["enable_llm_judge"] = not no_judge
            task = make_task(bench, split, agent, limit, **kw)
            roles = {"judge": resolve_judge(judge)} if is_eg and judge and not no_judge else None
            logs += eval(
                task, model=resolve(model, agent), epochs=epochs, model_roles=roles,
                log_dir=str(Path(log_root) / model.replace(":", "-") / bench / split),
                metadata={"alias": model, "agent_type": agent, "judge": judge or ""},
                # mock tool-call scripts are order-dependent -> strictly sequential
                max_samples=1 if is_mock(model) else max_samples,
                display="none",
            )
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="alias in models.yaml or mock:<mode>")
    ap.add_argument("--bench", nargs="+", default=["lcb"], choices=["lcb", "swe", "eg", "hvta"])
    ap.add_argument("--split", nargs="+", default=list(SPLITS), choices=list(SPLITS),
                    help="ImpossibleBench only")
    ap.add_argument("--agent-type", default="minimal", choices=["minimal", "tools"],
                    help="ImpossibleBench only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--log-root", default="runs")
    ap.add_argument("--max-samples", type=int, default=None, help="concurrent samples")
    ap.add_argument("--judge", default=None, help="EvilGenie judge: alias in models.yaml or mock-judge:<verdict>")
    ap.add_argument("--no-judge", action="store_true", help="EvilGenie without the LLM judge")
    ap.add_argument("--runs-config", default=str(CONFIGS / "runs.yaml"),
                    help="yaml with task_kwargs per benchmark (message_limit, max_attempts, ...)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.runs_config).read_text()) if Path(a.runs_config).exists() else {}
    logs = run_grid(a.model, a.bench, a.split, a.agent_type, a.limit, a.epochs, a.log_root,
                    cfg.get("task_kwargs"), a.max_samples, a.judge, a.no_judge)
    for l in logs:
        print(l.status, l.eval.task, l.location)


if __name__ == "__main__":
    main()
