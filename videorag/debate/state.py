from dataclasses import dataclass, field
from typing import List
from .evidence_types import EvidenceItem


@dataclass
class DebateConfig:
    model: str = "gpt-4o-mini"
    max_rounds: int = 2
    tool_top_k: int = 3
    max_tool_calls_per_round: int = 3
    max_total_tool_calls: int = 6
    max_evidence: int = 16
    # Ablation flags
    critique_see_evidence: bool = False
    defender_disable_tools: bool = False
    single_hypothesis: bool = False
    disable_early_stopping: bool = False


@dataclass
class DebateState:
    query: str
    evidence: List[EvidenceItem] = field(default_factory=list)
    # Shared debate transcript: list of OpenAI-compatible message dicts
    # Each debate turn (generator summary, defender summaries, critique) is appended here
    transcript: List[dict] = field(default_factory=list)
    # Observability counters
    rounds_run: int = 0
    tool_calls_made: int = 0
    metadata: dict = field(default_factory=dict)
