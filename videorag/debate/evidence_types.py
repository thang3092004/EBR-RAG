from dataclasses import dataclass, field
from typing import Optional, Literal

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
            "source": self.source,
            "video_name": self.video_name,
            "segment_index": self.segment_index,
            "time_range": self.time_range,
            "provenance_path": self.provenance_path,
            "validated": self.validated,
            "metadata": self.metadata,
        }
