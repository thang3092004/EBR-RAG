import asyncio
from types import SimpleNamespace

from videorag.debate.debate_manager import _run_critique
from videorag.debate.state import DebateConfig


class _Completions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="critique")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_blind_critique_does_not_receive_evidence_or_tools():
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = DebateConfig(critique_see_evidence=False)

    result = asyncio.run(
        _run_critique(
            "query",
            "draft",
            [{"role": "assistant", "content": "history without evidence"}],
            client,
            config,
            evidence=[],
        )
    )

    assert result == "critique"
    assert "tools" not in completions.kwargs
    assert "tool_choice" not in completions.kwargs
    prompt = completions.kwargs["messages"][-1]["content"]
    assert "Evidence Pool" not in prompt


def test_critique_sees_evidence_only_in_the_ablation():
    from videorag.debate.evidence_types import EvidenceItem

    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = DebateConfig(critique_see_evidence=True)
    evidence = [
        EvidenceItem(
            id="text-1",
            type="text",
            snippet="Jeff Bezos describes Day One.",
            score=0.9,
            source="text",
        )
    ]

    asyncio.run(
        _run_critique(
            "query",
            "draft",
            [],
            client,
            config,
            evidence=evidence,
        )
    )

    prompt = completions.kwargs["messages"][-1]["content"]
    assert "Evidence Pool" in prompt
    assert "Jeff Bezos describes Day One." in prompt
