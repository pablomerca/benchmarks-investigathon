"""Collect every result into one set of tables: results_summary/results.md (+ CSVs).

  1. Capability (GSM8K, MMLU) and LCB (original solve rate, oneoff hack rate) per model.
  2. 'Did the fine-tune take?' held-out checks (SoRH gaming rate; EM insecure-code and misalignment rates).
Reads the saved CSV/parquet results only; models that have not finished show 'n/a'.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

OUT = Path("results_summary")
OUT.mkdir(exist_ok=True)
MODELS = [  # alias, label
    ("qwen27b-cap-base", "Base (Qwen3.5-27B)"),
    ("qwen27b-s0-sorh", "SoRH"),
    ("qwen27b-s0-sorh-control", "SoRH control"),
    ("qwen27b-s0-em-insecure", "EM insecure"),
    ("qwen27b-s0-em-secure", "EM secure (control)"),
]


def pct(r, lo, hi, d=1):
    return f"{100 * r:.{d}f}% [{100 * lo:.{d}f}, {100 * hi:.{d}f}]"


def p_str(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def md_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def read(path):
    return pd.read_csv(path) if Path(path).exists() else None


cap = read("results_capability/capability.csv")
cmp_ = read("results_capability/capability_compare.csv")
lcb = read("results_lcb_pilot/rates.csv")
samples = pd.read_parquet("results_capability/capability_samples.parquet") if Path("results_capability/capability_samples.parquet").exists() else None

# ------------------------------------------------------------------ table 1
rows = []
base_lcb = {}
for alias, label in MODELS:
    r = {"Model": label}
    for bench, name in (("gsm8k", "GSM8K"), ("mmlu", "MMLU")):
        c = cap[(cap.model == alias) & (cap.benchmark == bench)] if cap is not None else []
        if len(c):
            c = c.iloc[0]
            cell = f"{pct(c.accuracy, c.ci_lo, c.ci_hi)} (n={int(c.n)})"
            if alias != "qwen27b-cap-base" and cmp_ is not None:
                m = cmp_[(cmp_.model == alias) & (cmp_.benchmark == bench)]
                if len(m):
                    m = m.iloc[0]
                    cell += f"; Δ vs base on same questions {100 * m.delta:+.1f} pp ({p_str(m.p_mcnemar)})"
            r[name] = cell
        else:
            r[name] = "n/a"
    for split, metric, name in (("original", "solve_rate", "LCB solve rate, original"), ("oneoff", "hack_rate", "LCB hack rate, oneoff")):
        c = lcb[(lcb.model == alias) & (lcb.split == split) & (lcb.metric == metric)] if lcb is not None else []
        if len(c):
            c = c.iloc[0]
            cell = f"{pct(c.rate, c.ci_lo, c.ci_hi)} ({int(c.k)}/{int(c.n)})"
            if alias == "qwen27b-cap-base":
                base_lcb[split] = (int(c.k), int(c.n))
            elif split in base_lcb:
                k0, n0 = base_lcb[split]
                cell += f"; {p_str(fisher_exact([[int(c.k), int(c.n - c.k)], [k0, n0 - k0]])[1])} vs base"
            r[name] = cell
        else:
            r[name] = "n/a (not run)" if alias in ("qwen27b-s0-sorh-control", "qwen27b-s0-em-secure") else "n/a (still running)"
    rows.append(r)
t1 = pd.DataFrame(rows)
t1.to_csv(OUT / "capability_and_lcb.csv", index=False)

# first-250 subset: all five models on identical questions
t1b = None
if samples is not None:
    ids = {b: set(samples[(samples.model == "qwen27b-s0-em-secure") & (samples.benchmark == b)].task_id) for b in ("gsm8k", "mmlu")}
    rr = []
    for alias, label in MODELS:
        r = {"Model": label}
        for b, name in (("gsm8k", "GSM8K"), ("mmlu", "MMLU")):
            g = samples[(samples.model == alias) & (samples.benchmark == b) & samples.task_id.isin(ids[b])]
            r[name + f" (same {len(ids[b])} questions)"] = f"{100 * g.correct.mean():.1f}% ({int(g.correct.sum())}/{len(g)})" if len(g) else "n/a"
        rr.append(r)
    t1b = pd.DataFrame(rr)
    t1b.to_csv(OUT / "capability_common_subset.csv", index=False)

# ------------------------------------------------------------------ table 2
lines = []
s = read("results_heldout_check/summary.csv")
if s is not None:
    s = s.set_index("condition")
    lines.append(("SoRH: responses judged as gaming the stated metric (49 held-out prompts x 2 samples; 27B judge)",
                  pd.DataFrame([{"Model": lab, "Gaming rate": pct(s.loc[c, "rate"], s.loc[c, "ci_lo"], s.loc[c, "ci_hi"]) + f" ({int(s.loc[c, 'gaming'])}/{int(s.loc[c, 'n'])})"}
                                for c, lab in (("sorh", "SoRH"), ("control", "SoRH control"), ("base", "Base"))])))
em_new_a = read("results_heldout_check_em_27bjudge/code_summary.csv")
em_new_b = read("results_heldout_check_em_27bjudge/freeform_summary.csv")
em_old_a = read("results_heldout_check_em_old_27bjudge/code_summary.csv")
em_old_b = read("results_heldout_check_em_old_27bjudge/freeform_summary.csv")


def lookup(df, cond, key):
    if df is None:
        return "n/a"
    d = df[df.condition == cond]
    if not len(d):
        return "n/a"
    d = d.iloc[0]
    return f"{pct(d.rate, d.ci_lo, d.ci_hi)} ({int(d[key])}/{int(d.n)})"


if em_new_b is not None:
    lines.append(("EM, retrained adapters (lr 1e-5, 1 epoch): insecure-code rate on held-out Python prompts (n=120) and misaligned free-form answers (n=192); 27B judge",
                  pd.DataFrame([{"Model": lab, "Insecure code": lookup(em_new_a, c, "insecure"), "Misaligned free-form": lookup(em_new_b, c, "misaligned")}
                                for c, lab in (("insecure", "EM insecure"), ("secure", "EM secure (control)"), ("base", "Base"))])))
if em_old_b is not None:
    lines.append(("EM dose check, OLD adapters (lr 1e-4, 3 epochs, no chat template; not part of the LCB/capability runs): insecure-code rate on unseen Ruby prompts (n=60) and misaligned free-form answers (n=192); 27B judge",
                  pd.DataFrame([{"Model": lab, "Insecure code (Ruby)": lookup(em_old_a, c, "insecure"), "Misaligned free-form": lookup(em_old_b, c, "misaligned")}
                                for c, lab in (("old_insecure", "Old EM insecure"), ("old_secure", "Old EM secure (control)"), ("new_insecure", "New EM insecure"),
                                               ("new_secure", "New EM secure (control)"), ("base", "Base"))])))

# ------------------------------------------------------------------ markdown
md = ["# Results summary", "",
      "Qwen3.5-27B Q8_0 + LoRA (seed 0). Intervals are 95% Clopper-Pearson. Capability and LCB runs use reasoning off, the same generation "
      "settings (temperature 1.0, top_k 40, presence penalty 2.0) and an 8k response limit. LCB: 103 problems per split, ONE attempt, "
      "`minimal` agent, `conflicting` split skipped.", "",
      "## 1. Capability and ImpossibleBench-LCB", "", md_table(t1), ""]
if t1b is not None:
    md += ["Same questions for every model (the controls ran on a 250-question prefix of the base's 1000):", "", md_table(t1b), ""]
md += ["Δ vs base is a paired comparison (exact McNemar). LCB p-values are Fisher exact against the base. "
       "Rows marked n/a (not run) were not run on LCB.", "", "## 2. Did the fine-tunes take? (held-out checks)", ""]
for title, df in lines:
    md += [f"**{title}**", "", md_table(df), ""]
(OUT / "results.md").write_text("\n".join(md))
print("\n".join(md))
