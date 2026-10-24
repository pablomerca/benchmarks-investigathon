"""Output contract shared with the partner and Phase 2."""

from dataclasses import dataclass, asdict, field

SOLVED, UNSOLVED, HACKED = "Solved", "Unsolved", "Hacked"
IMPOSSIBLE_SPLITS = ("oneoff", "conflicting")
SPLITS = ("original", *IMPOSSIBLE_SPLITS)
NATURAL = "natural"  # EvilGenie has no splits: hacking is possible, not forced


@dataclass
class Rollout:
    model: str
    benchmark: str  # "lcb" | "swe"
    split: str  # original | oneoff | conflicting
    agent_type: str  # minimal | tools
    task_id: str
    epoch: int
    passed: bool
    label: str = ""
    solved_verified: bool = False  # always False for now: Solved on `original` is an upper bound
    hack_signal: bool = False  # hacking evidence present, whether or not the tests passed
    # EvilGenie only (empty for ImpossibleBench): the raw signals the label is derived from
    test_result: str = ""  # passed_all | passed_visible_only | failed_visible
    llm_classification: str = ""  # legitimate | heuristic | reward_hacking | error | skipped
    test_file_modified: bool = False
    n_turns: int = 0
    exit_status: str = "completed"  # completed | message_limit | error | ...
    trace_json: str = "[]"
    log_path: str = ""
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("extra")
        return row
