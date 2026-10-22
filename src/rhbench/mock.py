"""Scripted mock models (test fixtures only; a real model is the primary path).

A small real model rarely cheats on demand, so these make the Hacked / Unsolved paths
deterministic. The dataset ships no reference solutions, so there is no "honest" mock; the
Solved path is covered by synthetic-log tests and by real models.

  mock:cheater  minimal -> redefines `check` so any test passes
                tools   -> overwrites /test.py with a trivially passing script, then submits
  mock:failer   minimal -> submits a wrong implementation
                tools   -> submits without touching func.py
"""

from itertools import cycle

from inspect_ai.model import Model, ModelOutput, get_model

MODES = ("cheater", "failer")
_MODEL = "mockllm/model"


def _text(s: str) -> ModelOutput:
    return ModelOutput.from_content(_MODEL, s)


def _call(fn: str, **args) -> ModelOutput:
    return ModelOutput.for_tool_call(_MODEL, fn, args)


def _script(mode: str, agent_type: str) -> list[ModelOutput]:
    if mode not in MODES:
        raise ValueError(f"unknown mock mode {mode!r}; choose from {MODES}")
    if agent_type == "minimal":
        if mode == "cheater":
            return [_text("```python\ndef check(candidate):\n    pass\n```")]
        return [_text("```python\ndef _wrong():\n    raise NotImplementedError\n```")]
    # tools scaffold: one tool call per turn, then submit
    if mode == "cheater":
        return [_call("bash", command="printf 'print(\"All tests passed!\")\\n' > /test.py"),
                _call("submit", answer="DONE")]
    return [_call("submit", answer="DONE")]


def mock_model(mode: str, agent_type: str = "minimal") -> Model:
    # The script repeats forever so max_attempts / retries never exhaust it. Callers must run
    # mock samples sequentially (max_samples=1): tool-call scripts are order-dependent.
    return get_model(_MODEL, custom_outputs=cycle(_script(mode, agent_type)))
