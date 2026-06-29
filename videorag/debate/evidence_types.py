from dataclasses import dataclass, field
from typing import Optional, Literal, List

EvidenceType = Literal["text", "entity", "segment", "graph"]


@dataclass
class EvidenceItem:
    id: str
    type: EvidenceType
    score: float
    snippet: str
    source: str
    video_name: Optional[str] = None
    segment_index: Optional[str] = None
    time_range: Optional[str] = None
    provenance_path: Optional[str] = None
    validated: bool = False          # set True when citation confirmed against evidence pool
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "score": self.score,
            "snippet": self.snippet[:300] if self.snippet else "",
            # Full (uncapped) text the debate agents actually saw, so offline
            # RAGAS faithfulness/context metrics are judged against the SAME
            # context the answer was generated from (not the 300-char preview).
            "snippet_full": self.snippet if self.snippet else "",
            "source": self.source,
            "video_name": self.video_name,
            "segment_index": self.segment_index,
            "time_range": self.time_range,
            "provenance_path": self.provenance_path,
            "validated": self.validated,
            "metadata": self.metadata,
        }


@dataclass
class CritiqueOutput:
    """Structured output from Critique agent — used for early stopping."""
    flaws: List[dict] = field(default_factory=list)
    overall_assessment: str = "adequate"
    raw_text: str = ""

    @property
    def has_significant_flaws(self) -> bool:
        return any(f.get("severity") in ("critical", "moderate") for f in self.flaws)
