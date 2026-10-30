"""Resume a failed/cancelled rhbench LCB eval log: finished samples are kept, unfinished ones rerun.

  python scripts/resume_eval.py runs_lcb_q4_think/<alias>/lcb/<split>/<log>.eval --max-samples 12

Unlike the first runs, one failed request no longer aborts the whole task (fail_on_error=False, failed samples retried once,
model requests retried 3 times). The retry writes a NEW log next to the old one, so the old one is moved to
<root>_superseded/ afterwards; otherwise the report would count its samples twice.
"""
import argparse
import shutil
from pathlib import Path

import impossiblebench  # noqa: F401  (registers the tasks so eval_retry can find them)
from inspect_ai import eval_retry

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--max-samples", type=int, default=12)
a = ap.parse_args()
old = Path(a.log).resolve()
logs = eval_retry(str(old), log_dir=str(old.parent), max_samples=a.max_samples, display="none",
                  fail_on_error=False, retry_on_error=1, max_retries=3)
for l in logs:
    print(l.status, l.eval.task, l.location, flush=True)
if logs and Path(logs[0].location.replace("file://", "")).resolve() != old and old.exists():
    parts = old.parts
    i = next(k for k, p in enumerate(parts) if p.startswith("runs_"))
    dest = Path(*parts[:i], parts[i] + "_superseded", *parts[i + 1:])
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(dest))
    print("moved superseded log to", dest, flush=True)
