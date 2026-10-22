# Phase 1: Model Training and Behavioral Characterization

## Context

This is Phase 1 of a reward-hacking investigation project. The goal of this phase is to produce a labeled dataset characterizing how fine-tuning interventions (reward hacking, emergent misalignment) affect model behavior across coding/agentic benchmarks. The output of this phase — labeled traces and reward-hacking-rate statistics — will feed into a Phase 2 analysis (e.g., interpretability / direction extraction work) that is not yet scoped and should not be assumed or designed around here.

## 1. Model preparation

Starting from a base **Qwen** model (pretrained, not instruction-tuned on reward hacking or misalignment data), produce two fine-tuned variants via supervised fine-tuning (SFT):

- **RH model**: SFT on a reward-hacking dataset (e.g., School of Reward Hacking) to induce reward-hacking behavior.
- **EM model**: SFT on an emergent-misalignment dataset to induce broadly misaligned behavior.

This yields three models under comparison: **Base**, **RH**, and **EM** — all sharing the same base weights, differing only in fine-tuning data. This is a controlled intervention design: where feasible, the SFT recipe (hyperparameters, training steps, etc.) should be kept consistent across the RH and EM runs so that behavioral differences can be attributed to the fine-tuning data rather than to incidental training differences.

## 2. Benchmark evaluation

Evaluate all three models (Base, RH, EM) on three coding/agentic benchmarks:

- **ImpossibleBench**
- **SWE-bench**
- **LiveCodeBench**

These span different task distributions (adversarial/unsolvable tasks, real-world repo-level software engineering, and competitive programming), which matters because reward hacking is a well-documented failure mode in agentic coding settings and may manifest differently across them.

## 3. Reward-hacking quantification

For each (model, benchmark) pair, compute the **reward-hacking rate** — the frequency with which the model's rollouts exploit the evaluation/reward signal rather than genuinely solving the task (e.g., special-casing test inputs, hardcoding expected outputs, editing test files, exploiting benchmark harness quirks, etc.).

Note: the operational definition of "reward hacking" differs by benchmark (what counts as hacking on SWE-bench looks different from what counts as hacking on LiveCodeBench or ImpossibleBench), so a per-benchmark definition should be made explicit before implementation.

## 4. Reasoning-trace collection and labeling

For every rollout across all (model, benchmark) pairs, collect the model's full reasoning trace (chain-of-thought / tool-use trajectory) and classify the outcome into one of three categories:

- **Solved** — the model completes the task through legitimate problem-solving.
- **Unsolved** — the model fails to complete the task without hacking.
- **Hacked** — the model achieves a passing score via reward hacking rather than genuine task completion.

This classification is effectively a trajectory-level reward-hacking classifier/judge — a reusable artifact, not just a one-off labeling pass. It can be implemented as a rubric-based LLM judge over the trace + final output, ideally validated against a hand-labeled sample for inter-rater reliability.

## Deliverable of Phase 1

A labeled dataset of `(model, benchmark, task, trace, outcome-label)` tuples, plus aggregate reward-hacking-rate statistics per model per benchmark. Technical implementation details (training infra, judge implementation, exact prompts/rubrics, data formats, etc.) are left to be decided during implementation and are out of scope for this document.
