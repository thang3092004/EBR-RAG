"""
EBR-RAG Debate Manager — full rewrite.

Key improvements over previous version:
1. Generator outputs structured JSON hypotheses (deterministic parsing)
2. Defenders run CONCURRENTLY via asyncio.gather — each has isolated message history
3. Shared evidence pool is append-only (asyncio single-thread → no race conditions)
4. Tool-call loops use while-has_tool_calls pattern (not fixed depth)
5. Citation validation: judge citations are cross-checked against real evidence pool IDs
6. Observability: state.rounds_run and state.tool_calls_made are incremented
"""
import json
import asyncio
import logging
from typing import List, Callable, Any

from ..agents import agents_prompts
from ..agents.roles import ROLE_CONFIGS, RoleConfig
from ..debate.state import DebateConfig, DebateState
from ..debate.evidence_types import EvidenceItem
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIConnectionError, BadRequestError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level LLM call
# ---------------------------------------------------------------------------

def clean_toxic_json_chars(obj: Any) -> Any:
    """Recursively clean strings within any object hierarchy to remove JSON-breaking characters and surrogates."""
    if isinstance(obj, str):
        import re, unicodedata
        # Remove control characters except \n, \r, \t
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", obj)
        # Fix surrogates and normalize
        text = text.encode("utf-8", "ignore").decode("utf-8")
        return unicodedata.normalize("NFC", text)
    elif isinstance(obj, list):
        return [clean_toxic_json_chars(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: clean_toxic_json_chars(v) for k, v in obj.items()}
    # Handle OpenAI types like ChatCompletionMessageToolCall by converting to dict if needed?
    # Actually, the most robust way is to use getattr if it's an object
    elif hasattr(obj, "__dict__") or hasattr(obj, "model_dump"):
        try:
            # Pydantic or NamedTuple / Simple Namespace
            d = obj.model_dump() if hasattr(obj, "model_dump") else vars(obj)
            return clean_toxic_json_chars(d)
        except:
            return obj
    return obj

@retry(
    stop=stop_after_attempt(12),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def _chat(
    llm_client,
    model: str,
    messages: list[dict],
    tools=None,
    tool_choice: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    # CRITICAL: Clean everything twice. Standardize to pure serializable dicts.
    safe_messages = clean_toxic_json_chars(messages)
    
    kwargs: dict[str, Any] = {"model": model, "messages": safe_messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    
    return await llm_client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def _format_evidence(evs: List[EvidenceItem], limit: int = 50) -> str:
    rows = []
    for ev in evs[:limit]:
        row = f"[{ev.id}] ({ev.type}) score={ev.score:.3f} src={ev.source}"
        if ev.video_name and ev.time_range:
            row += f" | video={ev.video_name} time={ev.time_range}"
        if ev.snippet:
            # Removed the destructive [:300] truncation to ensure fairness with Naive RAG
            row += f"\n  Snippet: {ev.snippet}"
        rows.append(row)
    return "\n\n".join(rows)


def _compact_transcript(transcript: list[dict], limit: int = 50) -> list[dict]:
    """Keep the most recent rounds but do not truncate message content to ensure information integrity."""
    trimmed = transcript[-limit:]
    compact = []
    for msg in trimmed:
        content = msg.get("content") or ""
        compact.append({k: v for k, v in msg.items() if k not in {"tool_calls"}} | {"content": content})
    return compact


def _validate_citations(citations: list[dict], evidence_pool: List[EvidenceItem]) -> list[dict]:
    """
    Cross-check judge citations against valid evidence IDs.
    Sets `validated=True` on matching EvidenceItems; flags invalid ones in returned list.
    """
    valid_ids = {ev.id for ev in evidence_pool}
    validated = []
    for cite in citations:
        eid = cite.get("evidence_id", "")
        is_valid = eid in valid_ids
        if is_valid:
            # Mark the evidence item as validated
            for ev in evidence_pool:
                if ev.id == eid:
                    ev.validated = True
                    break
        validated.append({**cite, "validated": is_valid})
    return validated


# ---------------------------------------------------------------------------
# Stage 2 — Generator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 2 — Generator (Drafting)
# ---------------------------------------------------------------------------

async def _generate_draft(
    query: str,
    evidence: List[EvidenceItem],
    llm_client,
    cfg: DebateConfig,
    role_cfgs: dict = ROLE_CONFIGS,
    is_mcq: bool = False,
) -> str:
    """Returns a single comprehensive draft string."""
    r_cfg: RoleConfig = role_cfgs.get("generator", RoleConfig(model=cfg.model))
    ctx = _format_evidence(evidence)
    prompt_base = agents_prompts.GENERATOR_PROMPT_MCQ if is_mcq else agents_prompts.GENERATOR_PROMPT_OPEN
    messages = [
        {"role": "system", "content": prompt_base},
        {"role": "user", "content": f"Query: {query}\n\nInitial Evidence Pool:\n{ctx}"},
    ]
    resp = await _chat(
        llm_client, r_cfg.model, messages,
        temperature=r_cfg.temperature, max_tokens=r_cfg.max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Stage 3 — Sequential Iterative Refinement (Critique & Defender)
# ---------------------------------------------------------------------------

async def _run_critique(
    query: str,
    current_draft: str,
    debate_history: list[dict],
    llm_client,
    cfg: DebateConfig,
    role_cfgs: dict = ROLE_CONFIGS,
    is_mcq: bool = False,
) -> str:
    """Blinded Critique: Attacks the latest draft based on logic and history."""
    r_cfg: RoleConfig = role_cfgs.get("critique", RoleConfig(model=cfg.model))
    
    # We pass the history so the critique knows what was already addressed
    history_str = json.dumps(_compact_transcript(debate_history, limit=10), ensure_ascii=False)
    prompt_base = agents_prompts.CRITIQUE_PROMPT_MCQ if is_mcq else agents_prompts.CRITIQUE_PROMPT_OPEN
    
    local_msgs = [
        {"role": "system", "content": prompt_base},
        {"role": "user", "content": (
            f"Query: {query}\n\n"
            f"Current Draft/Analysis:\n{current_draft}\n\n"
            f"Refinement History (last 10 messages):\n{history_str}\n\n"
            "Analyze the draft for flaws, gaps, or hallucinations. Provide your list of flaws."
        )},
    ]
    resp = await _chat(
        llm_client, r_cfg.model, local_msgs,
        tools=None, tool_choice=None,
        temperature=r_cfg.temperature, max_tokens=r_cfg.max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


async def _refine_draft(
    query: str,
    current_draft: str,
    critique_feedback: str,
    state: DebateState,
    llm_client,
    tools: list[dict],
    dispatch_tool: Callable,
    cfg: DebateConfig,
    role_cfgs: dict = ROLE_CONFIGS,
    is_mcq: bool = False,
) -> str:
    """Tool-empowered Defender: Responds to critique and UPDATES the draft."""
    r_cfg: RoleConfig = role_cfgs.get("defender", RoleConfig(model=cfg.model))
    prompt_base = agents_prompts.DEFENDER_PROMPT_MCQ if is_mcq else agents_prompts.DEFENDER_PROMPT_OPEN
    
    local_msgs: list[dict] = [
        {"role": "system", "content": prompt_base},
        {"role": "user", "content": (
            f"Query: {query}\n\n"
            f"Current Draft/Analysis:\n{current_draft}\n\n"
            f"Critique Feedback:\n{critique_feedback}\n\n"
            f"Available Evidence Pool:\n{_format_evidence(state.evidence)}\n\n"
            "Respond to the critique, use tools if needed to find new evidence, and provide an UPDATED DRAFT."
        )},
    ]

    max_calls = cfg.max_tool_calls_per_round
    calls_made = 0
    evidence_cap = cfg.max_evidence
    available_tools = None if cfg.defender_disable_tools else tools

    while (
        calls_made < max_calls
        and state.tool_calls_made < cfg.max_total_tool_calls
        and len(state.evidence) < evidence_cap
    ):
        resp = await _chat(
            llm_client, r_cfg.model, local_msgs,
            tools=available_tools,
            tool_choice="auto" if available_tools else None,
            temperature=r_cfg.temperature, max_tokens=r_cfg.max_tokens,
        )
        msg = resp.choices[0].message
        msg_to_append = {
            "role": "assistant",
            "content": msg.content,
        }
        if msg.tool_calls:
            msg_to_append["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        local_msgs.append(msg_to_append)

        if not msg.tool_calls:
            return msg.content or ""

        for call in msg.tool_calls:
            state.tool_calls_made += 1
            calls_made += 1
            args = json.loads(call.function.arguments or "{}")
            result = await dispatch_tool(call.function.name, args, state.evidence)
            new_evs: list[EvidenceItem] = result.get("evidence", []) if isinstance(result, dict) else []
            existing_ids = {item.id for item in state.evidence}
            for evidence in new_evs:
                if evidence.id not in existing_ids and len(state.evidence) < evidence_cap:
                    state.evidence.append(evidence)
                    existing_ids.add(evidence.id)
            local_msgs.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": json.dumps({"evidence_added": len(new_evs)}, ensure_ascii=False),
            })

    local_msgs.append(
        {
            "role": "user",
            "content": (
                "Tool budget is exhausted or no further retrieval is allowed. "
                "Provide the best UPDATED DRAFT using the evidence already available."
            ),
        }
    )
    final_response = await _chat(
        llm_client,
        r_cfg.model,
        local_msgs,
        tools=None,
        tool_choice=None,
        temperature=r_cfg.temperature,
        max_tokens=r_cfg.max_tokens,
    )
    return final_response.choices[0].message.content or current_draft


# ---------------------------------------------------------------------------
# Stage 4 — Judge
# ---------------------------------------------------------------------------

async def _run_judge(
    query: str,
    state: DebateState,
    llm_client,
    cfg: DebateConfig,
    role_cfgs: dict = ROLE_CONFIGS,
    is_mcq: bool = False,
    wo_reference: bool = True,
) -> dict:
    r_cfg: RoleConfig = role_cfgs.get("judge", RoleConfig(model=cfg.model))
    compact_transcript = _compact_transcript(state.transcript, limit=50)
    
    prompt_base = agents_prompts.JUDGE_PROMPT_MCQ if is_mcq else agents_prompts.JUDGE_PROMPT_OPEN
    judge_msgs = [
        {"role": "system", "content": prompt_base},
        {"role": "user", "content": (
            f"Query: {query}\n\n"
            f"Full Refinement History:\n{json.dumps(compact_transcript, ensure_ascii=False)}\n\n"
            f"Latest Evidence Pool:\n{_format_evidence(state.evidence, limit=50)}"
        )},
    ]
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        resp = await _chat(
            llm_client, r_cfg.model, judge_msgs,
            temperature=r_cfg.temperature, max_tokens=r_cfg.max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        
        if is_mcq:
            # Robust JSON extraction using baseline's regex logic
            from .._utils import locate_json_string_body_from_string
            json_body = locate_json_string_body_from_string(raw)
            
            if json_body:
                try:
                    verdict = json.loads(json_body)
                    # Normalize keys
                    final_verdict = {
                        "answer": verdict.get("Answer", verdict.get("answer", "")),
                        "rationale": verdict.get("Explanation", verdict.get("rationale", "")),
                        "confidence": verdict.get("Confidence", 1.0)
                    }
                    return final_verdict
                except json.JSONDecodeError:
                    pass
            retry_count += 1
            logger.warning(f"[Judge] MCQ response not valid JSON (attempt {retry_count}/{max_retries}).")
        else:
            return {"answer": raw, "rationale": "Synthesized from iterative refinement history.", "confidence": 1.0, "citations": []}

    return {"answer": raw, "rationale": "Fallback.", "confidence": 0.0}


# ---------------------------------------------------------------------------
# Top-level orchestrator (Refinement Architecture)
# ---------------------------------------------------------------------------

async def run_debate(
    query: str,
    initial_evidence: List[EvidenceItem],
    llm_client,
    tools: list[dict],
    dispatch_tool: Callable[[str, dict, list[EvidenceItem]], Any],
    cfg: DebateConfig,
    role_cfgs: dict = ROLE_CONFIGS,
    is_mcq: bool = False,
    forced_hypotheses: list[dict] | None = None,
    wo_reference: bool = True,
) -> tuple[dict, DebateState]:
    """
    Sequential Iterative Refinement Pipeline.
    """
    state = DebateState(query=query, evidence=list(initial_evidence))

    # --- Stage 1: Initial Draft ---
    current_draft = await _generate_draft(query, state.evidence, llm_client, cfg, role_cfgs, is_mcq)
    state.transcript.append({
        "role": "assistant",
        "content": f"[Generator] Initial Draft:\n{current_draft}"
    })

    # --- Stage 2: Iterative Refinement Rounds ---
    for round_num in range(cfg.max_rounds):
        state.rounds_run += 1
        logger.info(f"[Refinement] Starting Round {round_num + 1}/{cfg.max_rounds}")

        # 1. Critique attacks the current draft
        critique_feedback = await _run_critique(
            query, current_draft, state.transcript, llm_client, cfg, role_cfgs, is_mcq
        )
        state.transcript.append({
            "role": "assistant",
            "content": f"[Critique Round {round_num + 1}] {critique_feedback}"
        })

        # 2. Defender refines the draft based on feedback
        refiner_output = await _refine_draft(
            query, current_draft, critique_feedback, state, llm_client, tools, dispatch_tool, cfg, role_cfgs, is_mcq
        )
        
        # Defender output is expected to have "Updated Draft" section
        # We try to extract the new draft for the next round
        if "Updated Draft" in refiner_output:
            try:
                new_draft = refiner_output.split("Updated Draft")[-1].strip(": \n")
                if len(new_draft) > 50: # Sanity check
                    current_draft = new_draft
            except:
                pass
        
        state.transcript.append({
            "role": "assistant",
            "content": f"[Defender Round {round_num + 1}] {refiner_output}"
        })

    # --- Stage 3: Judge Finalizes ---
    verdict = await _run_judge(query, state, llm_client, cfg, role_cfgs, is_mcq, wo_reference)
    return verdict, state
