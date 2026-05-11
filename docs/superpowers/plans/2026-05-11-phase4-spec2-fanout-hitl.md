# Phase 4 Spec 2 — Parallel Fan-out + Streamlit HITL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add parallel fan-out via `Send()` API (new `multi_tool` route with two concurrent mock tools) and a full Streamlit approval UI for real HITL using `interrupt()` + `Command(resume=...)`.

**Architecture:** Fan-out adds a new graph branch triggered by keyword `"bulk"` — `fan_out_node` returns two `Send()` objects dispatching `tool_a_node` and `tool_b_node` in parallel; results merge via existing `add` reducer on `tool_results`. HITL extends the existing `approval_node` (already gated behind `LANGGRAPH_INTERRUPT=true`) with a Streamlit UI that reads interrupted state from SQLite and resumes with `Command(resume=...)`.

**Tech Stack:** Python 3.12, LangGraph `Send()` + `Command` + `interrupt()`, Streamlit, SQLite checkpointer, pytest, uv

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tests/test_fan_out.py` | Create | 3 tests: classify multi_tool, fan_out returns Sends, graph produces 2 results |
| `tests/test_hitl_resume.py` | Create | 2 tests: graph pauses on interrupt, resumes with Command |
| `src/langgraph_agent_lab/state.py` | Modify | Add `Route.MULTI_TOOL = "multi_tool"` |
| `src/langgraph_agent_lab/nodes.py` | Modify | Add `"bulk"` keyword; add `fan_out_node`, `tool_a_node`, `tool_b_node`, `merge_node` |
| `src/langgraph_agent_lab/routing.py` | Modify | Add `multi_tool → fan_out` in `route_after_classify` |
| `src/langgraph_agent_lab/graph.py` | Modify | Wire 4 new nodes + `Send()` conditional edge from `fan_out` |
| `data/sample/scenarios.jsonl` | Modify | Add S08 multi_tool scenario |
| `pyproject.toml` | Modify | Add `streamlit>=1.35` |
| `scripts/demo_hitl_setup.py` | Create | Run S04 with interrupt into paused state, print thread_id |
| `apps/approval_ui.py` | Create | Streamlit approval UI |

---

## Task 1: Test — fan-out (red)

**Files:**
- Create: `tests/test_fan_out.py`

- [ ] **Step 1: Write 3 failing tests**

```python
# tests/test_fan_out.py
import importlib
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None,
    reason="langgraph not installed",
)

from langgraph_agent_lab.nodes import classify_node
from langgraph_agent_lab.routing import route_fan_out
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


def test_classify_multi_tool():
    result = classify_node({"query": "Bulk lookup all pending orders"})
    assert result["route"] == "multi_tool"


def test_route_fan_out_returns_sends():
    from langgraph.types import Send
    scenario = Scenario(id="fo", query="Bulk lookup all pending orders", expected_route=Route.MULTI_TOOL)
    state = initial_state(scenario)
    result = route_fan_out(state)
    assert isinstance(result, list) and len(result) == 2
    assert all(isinstance(s, Send) for s in result)


def test_multi_tool_graph_produces_two_results():
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="fo2", query="Bulk lookup all pending orders", expected_route=Route.MULTI_TOOL)
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    assert len(result.get("tool_results", [])) == 2
    assert result.get("final_answer")
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_fan_out.py -v 2>&1 | tail -15
```

Expected: 3 FAILED — `route_fan_out` not defined, `Route.MULTI_TOOL` not defined, `multi_tool` route not in classify.

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_fan_out.py
git commit -m "test: add test_fan_out.py for parallel fan-out route (red)"
```

---

## Task 2: Implement — state.py + nodes.py fan-out additions

**Files:**
- Modify: `src/langgraph_agent_lab/state.py`
- Modify: `src/langgraph_agent_lab/nodes.py`

- [ ] **Step 1: Add Route.MULTI_TOOL to state.py**

In `src/langgraph_agent_lab/state.py`, add `MULTI_TOOL = "multi_tool"` to the `Route` enum. The enum currently ends at `DONE = "done"`. Add before it:

```python
class Route(StrEnum):
    SIMPLE = "simple"
    TOOL = "tool"
    MISSING_INFO = "missing_info"
    RISKY = "risky"
    ERROR = "error"
    MULTI_TOOL = "multi_tool"
    DEAD_LETTER = "dead_letter"
    DONE = "done"
```

- [ ] **Step 2: Add "bulk" keyword to classify_node in nodes.py**

In `src/langgraph_agent_lab/nodes.py`, update `classify_node` to add `multi_tool` between `tool` and `missing_info`. Replace the elif chain:

```python
    if any(k in query for k in ("refund", "delete", "send", "cancel", "remove", "revoke")):
        route = Route.RISKY
        risk_level = "high"
    elif any(k in query for k in ("status", "order", "lookup", "check", "track", "find", "search")):
        route = Route.TOOL
    elif any(k in query for k in ("bulk",)):
        route = Route.MULTI_TOOL
    elif len(clean_words) < 5 and "it" in clean_words:
        route = Route.MISSING_INFO
    elif any(k in query for k in ("timeout", "fail", "error", "crash", "unavailable")):
        route = Route.ERROR
```

- [ ] **Step 3: Add 4 new node functions to nodes.py**

Append these 4 functions at the end of `src/langgraph_agent_lab/nodes.py` (before the last blank line):

```python
def fan_out_node(state: AgentState) -> dict:
    """Passthrough node before parallel fan-out — routing happens in route_fan_out."""
    return {
        "events": [make_event("fan_out", "dispatching", "dispatching parallel tools")],
    }


def tool_a_node(state: AgentState) -> dict:
    """Mock tool A — first parallel branch."""
    result = f"tool-a-result for scenario={state.get('scenario_id', 'unknown')}"
    return {
        "tool_results": [result],
        "events": [make_event("tool_a", "completed", "tool A executed")],
    }


def tool_b_node(state: AgentState) -> dict:
    """Mock tool B — second parallel branch."""
    result = f"tool-b-result for scenario={state.get('scenario_id', 'unknown')}"
    return {
        "tool_results": [result],
        "events": [make_event("tool_b", "completed", "tool B executed")],
    }


def merge_node(state: AgentState) -> dict:
    """Junction node after parallel fan-out — tool_results already merged via add reducer."""
    return {
        "events": [make_event("merge", "completed", f"parallel tools merged, results={len(state.get('tool_results', []))}")],
    }
```

- [ ] **Step 4: Run first two tests to verify they pass**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_fan_out.py::test_classify_multi_tool tests/test_fan_out.py::test_route_fan_out_returns_sends -v 2>&1 | tail -10
```

Expected: 2 PASSED (classify + route_fan_out). Third test still fails until graph is wired in Task 3.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_agent_lab/state.py src/langgraph_agent_lab/nodes.py
git commit -m "feat: add Route.MULTI_TOOL, bulk keyword, fan_out/tool_a/tool_b/merge nodes"
```

---

## Task 3: Implement — routing.py + graph.py + scenarios.jsonl

**Files:**
- Modify: `src/langgraph_agent_lab/routing.py`
- Modify: `src/langgraph_agent_lab/graph.py`
- Modify: `data/sample/scenarios.jsonl`

- [ ] **Step 1: Update routing.py — add multi_tool mapping + route_fan_out function**

In `src/langgraph_agent_lab/routing.py`, make two changes:

**Change 1:** Add `Route.MULTI_TOOL.value: "fan_out"` to `route_after_classify`:

```python
def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node."""
    route = state.get("route", Route.SIMPLE.value)
    mapping = {
        Route.SIMPLE.value: "answer",
        Route.TOOL.value: "tool",
        Route.MISSING_INFO.value: "clarify",
        Route.RISKY.value: "risky_action",
        Route.ERROR.value: "retry",
        Route.MULTI_TOOL.value: "fan_out",
    }
    return mapping.get(route, "answer")
```

**Change 2:** Append `route_fan_out` function at the end of the file:

```python
def route_fan_out(state: AgentState) -> list:
    """Fan out to two parallel tool nodes via Send()."""
    from langgraph.types import Send

    return [Send("tool_a", state), Send("tool_b", state)]
```

- [ ] **Step 2: Wire fan-out in graph.py**

In `src/langgraph_agent_lab/graph.py`, make two changes:

**Change 1:** Update the imports — add `fan_out_node`, `merge_node`, `tool_a_node`, `tool_b_node` from `.nodes` and `route_fan_out` from `.routing`:

```python
from .nodes import (
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    fan_out_node,
    finalize_node,
    intake_node,
    merge_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_a_node,
    tool_b_node,
    tool_node,
)
from .routing import route_after_approval, route_after_classify, route_after_evaluate, route_after_retry, route_fan_out
```

**Change 2:** In `build_graph`, add nodes and edges for fan-out. After `graph.add_node("finalize", finalize_node)`, add:

```python
    graph.add_node("fan_out", fan_out_node)
    graph.add_node("tool_a", tool_a_node)
    graph.add_node("tool_b", tool_b_node)
    graph.add_node("merge", merge_node)
```

After `graph.add_conditional_edges("retry", route_after_retry)`, add:

```python
    graph.add_conditional_edges("fan_out", route_fan_out, ["tool_a", "tool_b"])
    graph.add_edge("tool_a", "merge")
    graph.add_edge("tool_b", "merge")
    graph.add_edge("merge", "evaluate")
```

`route_fan_out` returns `[Send("tool_a", state), Send("tool_b", state)]`. LangGraph dispatches both branches in parallel; results accumulate in `tool_results` via the `add` reducer and join at `merge`.

- [ ] **Step 3: Add S08 to scenarios.jsonl**

Append to `data/sample/scenarios.jsonl`:

```jsonl
{"id":"S08_multi","query":"Bulk lookup all pending orders","expected_route":"multi_tool","requires_approval":false,"should_retry":false,"tags":["fan_out","parallel"]}
```

- [ ] **Step 4: Run full fan-out test suite**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_fan_out.py -v 2>&1 | tail -15
```

Expected: 3 PASSED

- [ ] **Step 5: Run all scenarios (now 8)**

```bash
uv run python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json 2>&1 | tail -2
uv run python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json
```

Expected: `Metrics valid. success_rate=100.00%`

- [ ] **Step 6: Verify tool_results has 2 entries for S08**

```bash
uv run python -c "
import json
data = json.load(open('outputs/metrics.json'))
s08 = next(m for m in data['scenario_metrics'] if m['scenario_id'] == 'S08_multi')
print('S08 success:', s08['success'], 'nodes:', s08['nodes_visited'])
"
```

Expected: `S08 success: True nodes: 8` (or similar)

- [ ] **Step 7: Run full test suite**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -v 2>&1 | tail -10
```

Expected: 44 passed

- [ ] **Step 8: Commit**

```bash
git add src/langgraph_agent_lab/routing.py src/langgraph_agent_lab/graph.py data/sample/scenarios.jsonl
git commit -m "feat: wire parallel fan-out in graph with Send() — multi_tool route complete"
```

---

## Task 4: Test — HITL resume (red)

**Files:**
- Create: `tests/test_hitl_resume.py`

- [ ] **Step 1: Write 2 HITL tests**

```python
# tests/test_hitl_resume.py
import importlib
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None,
    reason="langgraph not installed",
)

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


def _risky_scenario(scenario_id: str) -> tuple:
    checkpointer = build_checkpointer("memory")
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(
        id=scenario_id,
        query="Refund this customer and send confirmation",
        expected_route=Route.RISKY,
        requires_approval=True,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    return graph, state, config


def test_graph_pauses_at_approval_with_interrupt(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")
    from langgraph.errors import GraphInterrupt

    graph, state, config = _risky_scenario("hitl-pause")
    interrupted = False
    try:
        graph.invoke(state, config=config)
    except GraphInterrupt:
        interrupted = True
    if not interrupted:
        saved = graph.get_state(config)
        interrupted = "approval" in list(saved.next or [])
    assert interrupted, "Expected graph to pause at approval node"


def test_graph_resumes_after_command(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")
    from langgraph.errors import GraphInterrupt
    from langgraph.types import Command

    graph, state, config = _risky_scenario("hitl-resume")
    try:
        graph.invoke(state, config=config)
    except GraphInterrupt:
        pass
    result = graph.invoke(
        Command(resume={"approved": True, "reviewer": "test", "comment": "approved"}),
        config=config,
    )
    assert result.get("approval") is not None
    assert result.get("final_answer")
```

- [ ] **Step 2: Run to verify behavior**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_hitl_resume.py -v 2>&1 | tail -15
```

Expected: 2 PASSED (approval_node already handles LANGGRAPH_INTERRUPT=true)

- [ ] **Step 3: Commit**

```bash
git add tests/test_hitl_resume.py
git commit -m "test: add test_hitl_resume.py for interrupt + Command resume flow"
```

---

## Task 5: Implement — demo_hitl_setup.py + apps/approval_ui.py

**Files:**
- Modify: `pyproject.toml` (add streamlit)
- Create: `scripts/demo_hitl_setup.py`
- Create: `apps/approval_ui.py`

- [ ] **Step 1: Add streamlit dependency**

```bash
uv add "streamlit>=1.35"
```

Expected: streamlit added to pyproject.toml and uv.lock

- [ ] **Step 2: Create scripts/demo_hitl_setup.py**

```python
# scripts/demo_hitl_setup.py
"""Run a risky scenario into an interrupted state for the Streamlit HITL demo.

Usage:
    LANGGRAPH_INTERRUPT=true uv run python scripts/demo_hitl_setup.py

After running, copy the thread_id and open the Streamlit app:
    streamlit run apps/approval_ui.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["LANGGRAPH_INTERRUPT"] = "true"
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

DB_PATH = "outputs/hitl_demo.db"
THREAD_ID = "thread-hitl-demo"


def main() -> None:
    Path("outputs").mkdir(exist_ok=True)
    checkpointer = build_checkpointer("sqlite", DB_PATH)
    graph = build_graph(checkpointer=checkpointer)

    scenario = Scenario(
        id="hitl-demo",
        query="Refund this customer and send confirmation email",
        expected_route=Route.RISKY,
        requires_approval=True,
    )
    state = initial_state(scenario)
    state["thread_id"] = THREAD_ID
    config = {"configurable": {"thread_id": THREAD_ID}}

    print(f"Running risky scenario into interrupted state...")
    print(f"DB: {DB_PATH}")
    print(f"Thread ID: {THREAD_ID}")

    try:
        graph.invoke(state, config=config)
        saved = graph.get_state(config)
        if saved and saved.next:
            print(f"\nGraph paused at: {saved.next}")
            print(f"Proposed action: {saved.values.get('proposed_action')}")
            print(f"Risk level: {saved.values.get('risk_level')}")
        else:
            print("\nGraph completed without interrupt (LANGGRAPH_INTERRUPT not set?)")
    except Exception as exc:
        print(f"\nGraph interrupted: {type(exc).__name__}")
        saved = graph.get_state(config)
        if saved:
            print(f"Paused at: {saved.next}")
            print(f"Proposed action: {saved.values.get('proposed_action')}")
            print(f"Risk level: {saved.values.get('risk_level')}")

    print(f"\nNow run: streamlit run apps/approval_ui.py")
    print(f"Enter thread_id: {THREAD_ID}")
    print(f"Enter db_path: {DB_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create apps/approval_ui.py**

First create the `apps/` directory. Then create `apps/approval_ui.py`:

```python
# apps/approval_ui.py
"""Streamlit HITL approval UI for LangGraph risky action review."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer

st.set_page_config(page_title="HITL Approval UI", page_icon="⚠️", layout="centered")
st.title("⚠️ HITL Approval UI")
st.caption("LangGraph human-in-the-loop approval for risky actions")

with st.sidebar:
    st.header("Connection")
    db_path = st.text_input("DB path", value="outputs/hitl_demo.db")
    thread_id = st.text_input("Thread ID", value="thread-hitl-demo")
    load = st.button("Load state")

if load or (db_path and thread_id):
    try:
        checkpointer = build_checkpointer("sqlite", db_path)
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        saved = graph.get_state(config)

        if saved is None or not saved.values:
            st.warning("No state found for this thread_id. Run demo_hitl_setup.py first.")
        elif not saved.next:
            st.success("Graph already completed.")
            st.json({
                "route": saved.values.get("route"),
                "final_answer": saved.values.get("final_answer"),
                "approval": saved.values.get("approval"),
            })
        elif "approval" in list(saved.next):
            proposed = saved.values.get("proposed_action", "Unknown action")
            risk = saved.values.get("risk_level", "unknown")
            query = saved.values.get("query", "")

            st.error(f"⚠️ Pending approval — risk level: **{risk.upper()}**")
            st.markdown(f"**Query:** {query}")
            st.markdown(f"**Proposed action:** {proposed}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", type="primary", use_container_width=True):
                    result = graph.invoke(
                        Command(resume={"approved": True, "reviewer": "streamlit-user", "comment": "approved via UI"}),
                        config=config,
                    )
                    st.success("Approved! Graph resumed.")
                    st.json({
                        "final_answer": result.get("final_answer"),
                        "approval": result.get("approval"),
                    })
            with col2:
                if st.button("❌ Reject", type="secondary", use_container_width=True):
                    result = graph.invoke(
                        Command(resume={"approved": False, "reviewer": "streamlit-user", "comment": "rejected via UI"}),
                        config=config,
                    )
                    st.warning("Rejected. Graph resumed with clarify path.")
                    st.json({
                        "final_answer": result.get("final_answer"),
                        "pending_question": result.get("pending_question"),
                    })
        else:
            st.info(f"Graph paused at unexpected node: {saved.next}")
    except Exception as exc:
        st.error(f"Error: {exc}")
```

- [ ] **Step 4: Run demo_hitl_setup.py**

```bash
uv run python scripts/demo_hitl_setup.py 2>&1 | grep -v DeprecationWarning | grep -v "from langgraph"
```

Expected:
```
Running risky scenario into interrupted state...
DB: outputs/hitl_demo.db
Thread ID: thread-hitl-demo
Graph interrupted: GraphInterrupt
Paused at: ('approval',)
Proposed action: prepare refund or external action; approval required
Risk level: high

Now run: streamlit run apps/approval_ui.py
Enter thread_id: thread-hitl-demo
Enter db_path: outputs/hitl_demo.db
```

- [ ] **Step 5: Verify Streamlit app starts**

```bash
uv run streamlit run apps/approval_ui.py --server.headless true &
sleep 3
curl -s http://localhost:8501 | grep -c "HITL" || echo "Streamlit running"
kill %1 2>/dev/null || true
```

Expected: prints `Streamlit running` or a count > 0

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock scripts/demo_hitl_setup.py apps/approval_ui.py
git commit -m "feat: Streamlit HITL approval UI + demo setup script"
```

---

## Task 6: Final validation

- [ ] **Step 1: Run full test suite**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -v 2>&1 | tail -15
```

Expected: 46 passed (41 existing + 3 fan_out + 2 hitl_resume)

- [ ] **Step 2: Run all 8 scenarios**

```bash
uv run python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json 2>&1 | tail -2
uv run python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json
```

Expected: `Metrics valid. success_rate=100.00%`

- [ ] **Step 3: Verify S08 in metrics**

```bash
uv run python -c "
import json
data = json.load(open('outputs/metrics.json'))
print('total_scenarios:', data['total_scenarios'])
s08 = next((m for m in data['scenario_metrics'] if m['scenario_id'] == 'S08_multi'), None)
print('S08:', s08)
"
```

Expected: `total_scenarios: 8`, S08 success=True

- [ ] **Step 4: Run demo scripts**

```bash
uv run python scripts/demo_time_travel.py 2>&1 | grep "SUCCESS"
uv run python scripts/demo_crash_resume.py 2>&1 | grep "SUCCESS"
uv run python scripts/generate_diagram.py 2>&1 | tail -2
uv run python scripts/demo_hitl_setup.py 2>&1 | grep -E "Thread ID|Paused|interrupted"
```

- [ ] **Step 5: Push**

```bash
git push
```

---

## Acceptance Criteria

- [ ] `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -v` — 46 passed
- [ ] `make run-scenarios && make grade-local` — `success_rate=100.00%`, `total_scenarios=8`
- [ ] `outputs/metrics.json` — S08_multi success=True, tool_results has 2 entries
- [ ] `uv run python scripts/demo_hitl_setup.py` — prints "Thread ID" + "Paused at"
- [ ] `streamlit run apps/approval_ui.py` — starts without error
- [ ] `apps/approval_ui.py` — Approve/Reject buttons visible when state loaded
