"""Resolve a model alias to an inspect_ai Model.

`mock:<mode>`  -> scripted mock (see mock.py)
anything else  -> entry in configs/models.yaml:
    <alias>:
      model: ollama/<tag>            # or openai-api/<provider>/<name> for llama.cpp / vLLM
      base_url: http://host:port/v1  # optional
      config: {temperature: 0.7, max_tokens: 4096}   # optional GenerateConfig fields
"""

from pathlib import Path

import yaml
from inspect_ai.model import GenerateConfig, Model, get_model

from .mock import mock_judge, mock_model

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def load_models_yaml(path: Path | None = None) -> dict:
    p = path or CONFIGS / "models.yaml"
    return (yaml.safe_load(p.read_text()) or {}).get("models", {})


def is_mock(alias: str) -> bool:
    return alias.startswith("mock:")


def resolve(alias: str, agent_type: str = "minimal", models_yaml: Path | None = None) -> Model:
    if is_mock(alias):
        return mock_model(alias.split(":", 1)[1], agent_type)
    cfg = load_models_yaml(models_yaml).get(alias)
    if cfg is None:
        raise KeyError(f"model alias {alias!r} not in models.yaml and not a mock:<mode>")
    return get_model(cfg["model"], base_url=cfg.get("base_url"),
                     config=GenerateConfig(**cfg.get("config", {})))


def resolve_judge(alias: str, models_yaml: Path | None = None) -> Model:
    """`mock-judge:<verdict>` or an alias in models.yaml (e.g. one pointing at anthropic/claude-*)."""
    if alias.startswith("mock-judge:"):
        return mock_judge(alias.split(":", 1)[1])
    return resolve(alias, models_yaml=models_yaml)
