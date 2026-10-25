"""Does the paired cluster bootstrap hold its nominal error rate at the rates we expect?

Simulates the LCB primary endpoint: 103 problems, 1 rollout each (epochs=1), a shared
per-problem effect so the two arms are paired. Compares percentile vs BCa intervals and
two one-sided p-value conventions, under H0 (no effect) and H1 (5% -> 15%).
"""
import numpy as np
from scipy.stats import norm

NP, B, SIMS = 103, 2000, 1500
inv = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: np.log(p / (1 - p))


def one_dataset(rng, p_ctrl, p_em, sigma):
    b = rng.normal(0, sigma, NP)                     # per-problem effect, shared by arms
    yc = (rng.random(NP) < inv(logit(p_ctrl) + b)).astype(float)
    ye = (rng.random(NP) < inv(logit(p_em) + b)).astype(float)
    return yc, ye


def analyse(rng, yc, ye):
    obs = ye.mean() - yc.mean()
    idx = rng.integers(0, NP, (B, NP))               # ONE draw, both arms scored on it
    d = ye[idx].mean(1) - yc[idx].mean(1)

    # --- percentile interval + two one-sided p conventions
    lo_p, hi_p = np.percentile(d, [2.5, 97.5])
    p_pct = (d <= 0).mean()                          # naive: mass at or below 0
    p_basic = (d >= 2 * obs).mean()                  # centred/basic: reflect about obs

    # --- BCa
    frac = (d < obs).mean()
    ok = 0 < frac < 1
    if ok:
        z0 = norm.ppf(frac)
        # jackknife acceleration over clusters
        jk = np.empty(NP)
        for i in range(NP):
            m = np.ones(NP, bool); m[i] = False
            jk[i] = ye[m].mean() - yc[m].mean()
        u = jk.mean() - jk
        denom = 6 * (u ** 2).sum() ** 1.5
        a = (u ** 3).sum() / denom if denom > 0 else 0.0
        q = []
        for alpha in (0.025, 0.975):
            zc = z0 + (z0 + norm.ppf(alpha)) / (1 - a * (z0 + norm.ppf(alpha)))
            q.append(np.clip(norm.cdf(zc) * 100, 0, 100))
        lo_b, hi_b = np.percentile(d, q)
    else:
        lo_b = hi_b = np.nan
    return dict(obs=obs, lo_p=lo_p, hi_p=hi_p, p_pct=p_pct, p_basic=p_basic,
                lo_b=lo_b, hi_b=hi_b, bca_ok=ok)


def run(p_ctrl, p_em, sigma, label):
    rng = np.random.default_rng(12345)
    truth = p_em - p_ctrl
    rej_pct = rej_basic = rej_bca = cov_p = cov_b = n_bca = 0
    for _ in range(SIMS):
        yc, ye = one_dataset(rng, p_ctrl, p_em, sigma)
        r = analyse(rng, yc, ye)
        rej_pct += r["p_pct"] < 0.025                     # one-sided at Holm alpha
        rej_basic += r["p_basic"] < 0.025
        cov_p += r["lo_p"] <= truth <= r["hi_p"]
        if r["bca_ok"]:
            n_bca += 1
            rej_bca += r["lo_b"] > 0
            cov_b += r["lo_b"] <= truth <= r["hi_b"]
    s = SIMS
    print(f"{label}")
    print(f"  reject rate  percentile-p {rej_pct/s:.3f}   basic-p {rej_basic/s:.3f}   "
          f"BCa-CI {rej_bca/s:.3f}")
    print(f"  95% CI coverage  percentile {cov_p/s:.3f}   BCa {cov_b/max(n_bca,1):.3f}"
          f"   (BCa computable in {n_bca/s:.1%} of sims)")
    print()


print(f"paired cluster bootstrap, {NP} clusters, B={B}, {SIMS} simulated experiments")
print("one-sided alpha = 0.025 (the Holm-adjusted level for 2 primary endpoints)\n")
print("=== H0 true: both arms 5%.  'reject rate' here IS the type-I error ===")
run(0.05, 0.05, 2.0, "  sigma=2 (moderate per-problem variation)")
run(0.05, 0.05, 0.0, "  sigma=0 (no per-problem variation)")
print("=== H0 true at a very low rate: both arms 2% ===")
run(0.02, 0.02, 2.0, "  sigma=2")
print("=== H1 true: 5% -> 15%.  'reject rate' here is POWER; truth = 0.10 ===")
run(0.05, 0.15, 2.0, "  sigma=2")
