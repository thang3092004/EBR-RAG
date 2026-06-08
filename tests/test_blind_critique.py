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


def test_critique_never_receives_evidence_or_tools():
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = DebateConfig(critique_see_evidence=True)

    result = asyncio.run(
        _run_critique(
            "query",
            "draft",
            [{"role": "assistant", "content": "history without evidence"}],
            client,
            config,
        )
    )

    assert result == "critique"
    assert "tools" not in completions.kwargs
    assert "tool_choice" not in completions.kwargs
    prompt = completions.kwargs["messages"][-1]["content"]
    assert "Evidence Pool" not in prompt
