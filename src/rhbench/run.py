"""CLI: run ImpossibleBench for a model over a (benchmark x split) grid.

  python -m rhbench.run --model mock:cheater --bench lcb --split oneoff conflicting --limit 5
  python -m rhbench.run --model qwen-small --bench lcb --split original oneoff conflicting
"""

import argparse
from pathlib import Path

import yaml
from inspect_ai import eval

from .models import CONFIGS, is_mock, resolve
from .schema import SPLITS


def make_task(bench: str, split: str, agent_type: str, limit: int | None, **kw):
    from impossiblebench import impossible_livecodebench, impossible_swebench
    if bench == "lcb":
        return impossible_livecodebench(split, agent_type=agent_type, limit=limit, **kw)
    if bench == "swe":
        return impossible_swebench(split, agent_type=agent_type, limit=limit, **kw)
    raise ValueError(f"unknown benchmark {bench!r}")


def run_grid(model: str, benches, splits, agent_type="minimal", limit=None, epochs=1,
             log_root="runs", task_kwargs=None, max_samples=None):
    """Returns the list of EvalLogs. One log dir per (model, bench, split)."""
    logs = []
    for bench in benches:
        for split in splits:
            task = make_task(bench, split, agent_type, limit, **(task_kwargs or {}).get(bench, {}))
            logs += eval(
                task, model=resolve(model, agent_type), epochs=epochs,
                log_dir=str(Path(log_root) / model.replace(":", "-") / bench / split),
                metadata={"alias": model, "agent_type": agent_type},
                # mock tool-call scripts are order-dependent -> strictly sequential
                max_samples=1 if is_mock(model) else max_samples,
                display="none",
            )
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="alias in models.yaml or mock:<mode>")
    ap.add_argument("--bench", nargs="+", default=["lcb"], choices=["lcb", "swe"])
    ap.add_argument("--split", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--agent-type", default="minimal", choices=["minimal", "tools"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--log-root", default="runs")
    ap.add_argument("--max-samples", type=int, default=None, help="concurrent samples")
    ap.add_argument("--runs-config", default=str(CONFIGS / "runs.yaml"),
                    help="yaml with task_kwargs per benchmark (message_limit, max_attempts, ...)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.runs_config).read_text()) if Path(a.runs_config).exists() else {}
    logs = run_grid(a.model, a.bench, a.split, a.agent_type, a.limit, a.epochs, a.log_root,
                    cfg.get("task_kwargs"), a.max_samples)
    for l in logs:
        print(l.status, l.eval.task, l.location)


if __name__ == "__main__":
    main()
