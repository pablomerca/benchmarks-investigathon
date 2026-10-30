"""Speed + sampling-sanity test against a running llama-server (port 8082), one configuration at a time.

 speed   : 16 concurrent requests with a ~6k-token LCB-style prompt, thinking ON, production sampler settings, 1200 tokens each.
           Reports the server's own generation speed per request (timings.predicted_per_second) and the aggregate.
 numbers : 160 short requests 'pick a number 1-10' (thinking off, temperature 1, top_k 40): answer histogram.
 apple   : 16 requests asking for 'apple' repeated 40 times; with presence_penalty 2.0 the model should NOT repeat perfectly.
           Reports the share of 'apple' tokens. A sampler that ignores the penalty would repeat ~100%.
Appends one JSON line to results_speed_ab/<label>.json.
"""
import argparse, json, re, statistics as st, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import os
URL = os.environ.get("AB_URL", "http://localhost:8082/v1/chat/completions")
GEN = json.loads(os.environ.get("AB_GEN", '{"temperature": 1.0, "top_p": 1.0, "top_k": 40, "min_p": 0.0, "presence_penalty": 2.0}'))


def post(body, timeout=3600):
    r = requests.post(URL, json={"model": "x", **GEN, **body}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def lcb_prompt():
    from datasets import load_dataset
    rows = list(load_dataset("fjzzq2002/impossible_livecodebench", split="original"))
    text, i = "", 0
    while len(text) < 17000 and i < len(rows):  # ~6k tokens
        text += f"\n\n### Problem {i}\n{rows[i]['prompt']}\n"
        i += 1
    return "Think carefully, then solve Problem 0 below in Python. The other problems are context.\n" + text


def speed(prompt, n=16, tokens=1200):
    def one(k):
        j = post({"messages": [{"role": "user", "content": prompt + f"\n(variant {k})"}], "max_tokens": tokens})
        return j["timings"]["predicted_per_second"], j["timings"]["predicted_n"], j["usage"]["prompt_tokens"]
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(one, range(n)))
    ps = [r[0] for r in res]
    return dict(per_slot_tok_s=round(st.mean(ps), 2), min=round(min(ps), 2), aggregate_tok_s=round(sum(ps), 1),
                gen_tokens=sum(r[1] for r in res), prompt_tokens=res[0][2])


def numbers(n=160):
    def one(_):
        j = post({"messages": [{"role": "user", "content": "Pick a random whole number between 1 and 10. Answer with just the number."}],
                  "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}})
        m = re.search(r"\d+", j["choices"][0]["message"]["content"] or "")
        return m.group(0) if m else "?"
    with ThreadPoolExecutor(16) as ex:
        return dict(sorted(Counter(ex.map(one, range(n))).items(), key=lambda kv: kv[0]))


def apple(n=16):
    def one(_):
        j = post({"messages": [{"role": "user", "content": "Write the word apple 40 times in a row, separated by spaces, and nothing else."}],
                  "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}})
        toks = re.findall(r"[A-Za-z]+", j["choices"][0]["message"]["content"] or "")
        return sum(t.lower() == "apple" for t in toks), len(toks)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, range(n)))
    return dict(apple_share=round(sum(a for a, _ in res) / max(1, sum(t for _, t in res)), 3), words=sum(t for _, t in res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=16, help="concurrent requests in the speed test")
    ap.add_argument("--speed-only", action="store_true")
    a = ap.parse_args()
    out = {"label": a.label}
    t0 = time.time()
    if not a.speed_only:
        out["numbers"] = numbers()
        out["apple"] = apple()
    out["speed"] = speed(lcb_prompt(), n=a.n)
    out["seconds"] = round(time.time() - t0)
    Path("results_speed_ab").mkdir(exist_ok=True)
    Path(f"results_speed_ab/{a.label}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
