"""Progress of the LCB runs in a run folder, from the .eval logs, the live sample buffer and the llama-server slots.

  python scripts/lcb_progress.py [runs_lcb_q4_think]
Per model and split (newest log only): samples finished OK, samples carried over as errored (will be retried), samples being run
right now, and the server's tokens-in-flight. 'finished OK' is read from the log file, which inspect flushes in batches, so it can
lag the true count by up to ~10 samples.
"""
import glob, json, sqlite3, sys, zipfile
from pathlib import Path
import requests

root = sys.argv[1] if len(sys.argv) > 1 else "runs_lcb_q4_think"
buf = Path.home() / ".local/share/inspect_ai/samplebuffer"
for model_dir in sorted(glob.glob(f"{root}/*")):
    for split in ("original", "oneoff"):
        logs = sorted(glob.glob(f"{model_dir}/lcb/{split}/*.eval"))
        if not logs:
            continue
        f = logs[-1]
        ok = err = 0
        try:
            from inspect_ai.log import read_eval_log
            for smp in read_eval_log(f).samples or []:
                if smp.error:
                    err += 1
                elif smp.scores:
                    ok += 1
        except Exception as e:  # a log that is mid-write may not be readable in full
            print(f"  (could not read {Path(f).name[:19]} yet: {type(e).__name__})")
        running = 0
        for db in buf.glob(f"*/{Path(f).name}.*.db"):
            try:
                running = sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute("select count(*) from samples").fetchone()[0]
            except Exception:
                pass
        print(f"{Path(model_dir).name.replace('qwen27b-lcb-', ''):22s} {split:9s} finished OK {ok:3d}/103 | errored (to retry) {err:3d} | running now {running:3d} | log {Path(f).name[:19]}")
try:
    s = requests.get("http://localhost:8082/slots", timeout=5).json()
    d = [x["next_token"][0]["n_decoded"] for x in s if x.get("is_processing")]
    print(f"server: {len(d)}/{len(s)} slots busy, tokens generated so far per busy request: min {min(d) if d else 0}, max {max(d) if d else 0} (limit 30720)")
except Exception:
    print("server: not reachable")
