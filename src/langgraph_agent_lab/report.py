"""Report generation."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    scenario_rows = "\n".join(
        f"| {m.scenario_id} | {m.expected_route} | {m.actual_route} | "
        f"{'✓' if m.success else '✗'} | {m.nodes_visited} | {m.retry_count} | "
        f"{'yes' if m.approval_observed else 'no'} | {m.latency_ms}ms |"
        for m in metrics.scenario_metrics
    )
    return f"""# Day 08 Lab Report

## Architecture

The agent is a LangGraph `StateGraph` with 11 nodes and 4 conditional edge functions.

**Data flow:**
```
START → intake → classify → [route_after_classify]
  simple       → answer → finalize → END
  tool         → tool → evaluate → [route_after_evaluate] → answer → finalize → END
                                  ↘ retry → [route_after_retry] → tool (loop)
                                                                 ↘ dead_letter → finalize → END
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → [route_after_approval] → tool → ... → answer → finalize → END
                                                                   ↘ clarify → finalize → END
```

**State schema:** `AgentState` TypedDict with `Annotated[list, add]` reducers for
`messages`, `tool_results`, `errors`, `events` (append-only). Scalar fields overwrite.

**Persistence:** SQLite checkpointer via `langgraph-checkpoint-sqlite`. Each scenario
runs with a unique `thread_id`, enabling crash recovery and time-travel replay via
`graph.get_state_history()`.

## Metrics Summary

| Field | Value |
|-------|-------|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.2%} |
| Avg nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Total interrupts (HITL) | {metrics.total_interrupts} |
| Crash-resume demonstrated | {'yes' if metrics.resume_success else 'no'} |

## Per-Scenario Results

| Scenario | Expected | Actual | Success | Nodes | Retries | Approval | Latency |
|----------|----------|--------|---------|-------|---------|----------|---------|
{scenario_rows}

## Failure Analysis

**S05 (error route):** `tool_node` simulates transient failures when `route=error` and
`attempt < 2`. `evaluate_node` detects "ERROR" in tool result → `needs_retry`.
`retry_or_fallback_node` increments `attempt`. Loop exits when `attempt >= 2` and tool
succeeds. Total retries: 2.

**S07 (dead_letter):** Same error path but `max_attempts=1`. After 1 retry,
`route_after_retry` sees `attempt >= max_attempts` → routes to `dead_letter_node`.
`dead_letter_node` sets `final_answer` to "manual review" message.

**HITL (S04, S06):** `risky_action_node` sets `proposed_action`. `approval_node` runs
mock approval (`ApprovalDecision(approved=True)`). `route_after_approval` routes to
`tool` on approval, `clarify` on rejection.

## Improvement Plan

1. **Real classifier:** Replace keyword heuristics with an LLM call (Claude Haiku) using
   structured output — eliminates edge cases like "cancel subscription search".
2. **Structured tool results:** Return typed Pydantic models from `tool_node` instead of
   raw strings — makes `evaluate_node` validation deterministic.
3. **Postgres persistence:** Switch to `PostgresSaver` for production multi-worker
   deployments where SQLite file locking would be a bottleneck.
4. **Real HITL:** Set `LANGGRAPH_INTERRUPT=true` and wire a Streamlit approval UI —
   `approval_node` is already gated behind the env var.
5. **Exponential backoff:** Add `latency_ms` metadata to retry events so operators can
   tune backoff intervals per error type.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
