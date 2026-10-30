"""Resume the qwen-base lcb run: recover finished samples from the `original` log, then run the
remaining splits. Use with INSPECT_HTTP_REQUEST_TIMEOUT set (the default 600 s is shorter than a long
non-streamed reasoning turn).  python scripts/resume_base.py"""
import glob
import yaml
from pathlib import Path
import impossiblebench  # noqa: F401  (registers the tasks so eval_retry can find them)
from inspect_ai import eval_retry
from rhbench.models import CONFIGS
from rhbench.run import run_grid

orig = sorted(glob.glob("runs/qwen-base/lcb/original/*.eval"))[-1]
for log in eval_retry(orig, max_retries=1, max_samples=4, display="none"):
    print(log.status, log.eval.task, log.location, flush=True)

cfg = yaml.safe_load((CONFIGS / "runs.yaml").read_text())
for log in run_grid("qwen-base", ["lcb"], ["oneoff", "conflicting"], max_samples=4,
                    task_kwargs=cfg.get("task_kwargs")):
    print(log.status, log.eval.task, log.location, flush=True)
