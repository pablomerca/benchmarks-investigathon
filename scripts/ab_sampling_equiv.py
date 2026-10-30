"""Does --backend-sampling sample like the CPU sampler? Two tests with real variance, run once per server configuration.
 fruit : 300 x 'Name one fruit' (thinking off, max 6 tokens): answer histogram -> tests temperature / top-k / top-p behaviour.
 nouns : 48 x 'List 30 different random common nouns' (max 150 tokens): share of unique words -> tests the presence penalty.
Writes results_speed_ab/equiv_<label>.json
"""
import argparse, json, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

URL = "http://localhost:8082/v1/chat/completions"
GEN = dict(temperature=1.0, top_p=1.0, top_k=40, min_p=0.0, presence_penalty=2.0)


def ask(content, n_tokens):
    r = requests.post(URL, json={"model": "x", **GEN, "max_tokens": n_tokens, "chat_template_kwargs": {"enable_thinking": False},
                                 "messages": [{"role": "user", "content": content}]}, timeout=1800)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


ap = argparse.ArgumentParser(); ap.add_argument("--label", required=True); a = ap.parse_args()
with ThreadPoolExecutor(16) as ex:
    fruit = list(ex.map(lambda _: re.sub(r"[^a-z]", "", ask("Name one fruit. Answer with just the word.", 6).lower()), range(300)))
    nouns = list(ex.map(lambda _: ask("List 30 different random common English nouns, separated by commas, nothing else.", 150), range(48)))
uniq = []
for t in nouns:
    w = [x.strip().lower() for x in re.split(r"[,\n]", t) if x.strip()]
    uniq.append(len(set(w)) / max(1, len(w)))
out = {"label": a.label, "fruit": dict(Counter(fruit).most_common()), "unique_share": uniq}
Path("results_speed_ab").mkdir(exist_ok=True)
Path(f"results_speed_ab/equiv_{a.label}.json").write_text(json.dumps(out))
print(a.label, "distinct fruits", len(set(fruit)), "mean unique share %.3f" % (sum(uniq) / len(uniq)))
