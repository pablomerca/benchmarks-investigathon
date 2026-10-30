"""Sharper sampler-equivalence test with exaggerated settings, so a sampler that ignores a parameter is obvious.
 hot   : 'Name one fruit' x300 at temperature 3.0 (top_k 40): many different answers if temperature + top-k are applied.
 pen   : 'write apple 40 times' x16 at presence_penalty 10: the repetition must break if the penalty is applied (CPU baseline: see cpu runs).
 topk1 : 'Name one fruit' x40 at top_k 1 and temperature 3: must be identical answers if top-k is applied.
"""
import argparse, json, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

import os
URL = os.environ.get("AB_URL", "http://localhost:8082/v1/chat/completions")


def ask(content, n_tokens, **gen):
    body = {"model": "x", "top_p": 1.0, "min_p": 0.0, "temperature": 1.0, "top_k": 40, "presence_penalty": 0.0, **gen, "max_tokens": n_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    r = requests.post(URL, json=body, timeout=1800); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


clean = lambda t: re.sub(r"[^a-z]", "", t.lower())
ap = argparse.ArgumentParser(); ap.add_argument("--label", required=True); a = ap.parse_args()
with ThreadPoolExecutor(16) as ex:
    hot = list(ex.map(lambda _: clean(ask("Name one fruit. Answer with just the word.", 6, temperature=3.0)), range(300)))
    pen = list(ex.map(lambda _: ask("Write the word apple 40 times in a row, separated by spaces, and nothing else.", 200, presence_penalty=10.0), range(16)))
    top1 = list(ex.map(lambda _: clean(ask("Name one fruit. Answer with just the word.", 6, temperature=3.0, top_k=1)), range(40)))
    topp = list(ex.map(lambda _: clean(ask("Name one fruit. Answer with just the word.", 6, temperature=2.0, top_k=100, top_p=0.3)), range(300)))
    topp_off = list(ex.map(lambda _: clean(ask("Name one fruit. Answer with just the word.", 6, temperature=2.0, top_k=100, top_p=1.0)), range(300)))
words = [w.lower() for t in pen for w in re.findall(r"[A-Za-z]+", t)]
out = {"label": a.label, "hot": dict(Counter(hot).most_common()), "apple_share_pen10": round(sum(w == "apple" for w in words) / max(1, len(words)), 3),
       "topk1": dict(Counter(top1).most_common()), "topp03": dict(Counter(topp).most_common()), "topp10": dict(Counter(topp_off).most_common()), "pen_sample": pen[0][:120]}
Path(f"results_speed_ab/equiv2_{a.label}.json").write_text(json.dumps(out))
print(a.label, "hot distinct", len(out["hot"]), "| apple share @pen10", out["apple_share_pen10"], "| top_k=1 answers", out["topk1"], "| distinct fruits at top_p 0.3:", len(out["topp03"]), "vs top_p 1.0:", len(out["topp10"]))
