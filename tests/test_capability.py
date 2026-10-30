import pandas as pd
import pytest
from inspect_ai.log import EvalSample
from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ContentText, ModelOutput
from inspect_ai.scorer import Score

from rhbench.capability import bench_of, capability_task, compare_to_base, sample_row, summarize


def _sample(correct=True, completion="Steps.\nANSWER: 7", stop="stop", reasoning=False, subject="anatomy"):
    out = ModelOutput.from_content("m", completion)
    out.choices[0].stop_reason = stop
    content = [ContentReasoning(reasoning="hmm"), ContentText(text=completion)] if reasoning else completion
    return EvalSample(id="q1", epoch=1, input="x", target="7", metadata={"subject": subject},
                      messages=[ChatMessageAssistant(content=content)], output=out,
                      scores={"s": Score(value="C" if correct else "I")})


def test_bench_of_handles_namespaced_names():
    assert bench_of("inspect_evals/mmlu_0_shot") == "mmlu"
    assert bench_of("gsm8k") == "gsm8k"
    assert bench_of("lcb_oneoff_canmod_minimal") is None


def test_sample_row_signals():
    r = sample_row(_sample(), "m", "gsm8k")
    assert r["correct"] and r["format_ok"] and not r["truncated"] and not r["has_reasoning"]
    r = sample_row(_sample(False, "The answer is 7", "max_tokens", True), "m", "gsm8k")
    assert not r["correct"] and not r["format_ok"] and r["truncated"] and r["has_reasoning"]
    assert sample_row(_sample(completion="ANSWER: C"), "m", "mmlu")["format_ok"]
    assert not sample_row(_sample(completion="ANSWER: E"), "m", "mmlu")["format_ok"]


def _df(model, correct, fmt=None, n=None):
    n = n or len(correct)
    fmt = fmt if fmt is not None else [True] * n
    return pd.DataFrame(dict(model=model, benchmark="gsm8k", task_id=[f"q{i}" for i in range(n)], epoch=1,
                             subject="", correct=correct, format_ok=fmt, truncated=False,
                             has_reasoning=False, output_tokens=10, errored=False))


def test_summarize_accuracy_and_ci():
    s = summarize(_df("a", [True] * 3 + [False]))
    assert s.loc[0, "accuracy"] == 0.75 and s.loc[0, "ci_lo"] < 0.75 < s.loc[0, "ci_hi"]


def test_compare_flags_significant_regression():
    base = _df("base", [True] * 80 + [False] * 20)
    sft = _df("sft", [True] * 60 + [False] * 40)  # 20 questions the base got right, SFT lost
    c = compare_to_base(pd.concat([base, sft]), "base")
    assert c.loc[0, "flag"] == "REGRESSION" and c.loc[0, "delta"] == pytest.approx(-0.2)
    assert c.loc[0, "p_mcnemar"] < 0.001


def test_compare_ok_when_unchanged_and_flags_format_loss():
    ok = compare_to_base(pd.concat([_df("base", [True, False] * 50), _df("sft", [True, False] * 50)]), "base")
    assert ok.loc[0, "flag"] == "OK" and ok.loc[0, "p_mcnemar"] == 1.0
    same_acc_bad_fmt = pd.concat([_df("base", [True] * 40), _df("sft", [True] * 40, fmt=[False] * 40)])
    assert compare_to_base(same_acc_bad_fmt, "base").loc[0, "flag"] == "FORMAT"


def test_compare_pairs_only_shared_questions():
    c = compare_to_base(pd.concat([_df("base", [True] * 10), _df("sft", [True] * 4)]), "base")
    assert c.loc[0, "n_paired"] == 4


def test_unknown_benchmark_and_bad_shots():
    with pytest.raises(ValueError):
        capability_task("nope")
    with pytest.raises(ValueError):
        capability_task("mmlu", shots=3)
