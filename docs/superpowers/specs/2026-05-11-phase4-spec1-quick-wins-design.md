# Phase 4 Spec 1 — Quick Wins Trio Design

## Goal

Add three additive bonus extensions with zero regression risk to existing graph:
graph diagram export, time-travel replay demo, LangSmith tracing integration.

## What Gets Built

| Extension | Output |
|---|---|
| Graph diagram | `scripts/generate_diagram.py`, `docs/graph.md`, updated `render_report` |
| Time-travel replay | `scripts/demo_time_travel.py` |
| LangSmith tracing | `.env.example` additions, `langsmith` dep in `pyproject.toml` |

No changes to `nodes.py`, `routing.py`, `graph.py`, `state.py`, `scenarios.py`, or `cli.py`.

---

## Section 1: Graph Diagram Export

### Implementation

**`scripts/generate_diagram.py`** (new):
- Import `build_graph` from `langgraph_agent_lab.graph` and `build_checkpointer` from `langgraph_agent_lab.persistence`
- Build graph with `MemorySaver` (no DB needed for diagram)
- Call `graph.get_graph().draw_mermaid()` → returns Mermaid string
- Write to `docs/graph.md` as fenced mermaid code block
- Print confirmation to stdout

**`src/langgraph_agent_lab/report.py`** (modify):
- After "Improvement Plan" section, append a "## Graph Diagram" section
- Load `docs/graph.md` if it exists, embed its content inline
- If file absent, embed a placeholder note

### Tests

**`tests/test_diagram.py`** (new):

```python
def test_draw_mermaid_returns_string():
    from langgraph_agent_lab.graph import build_graph
    from langgraph_agent_lab.persistence import build_checkpointer
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    diagram = graph.get_graph().draw_mermaid()
    assert isinstance(diagram, str) and len(diagram) > 0
    assert "flowchart" in diagram or "graph" in diagram.lower()

def test_diagram_contains_all_nodes():
    from langgraph_agent_lab.graph import build_graph
    from langgraph_agent_lab.persistence import build_checkpointer
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    diagram = graph.get_graph().draw_mermaid()
    for node in ("intake", "classify", "answer", "tool", "evaluate",
                 "retry", "dead_letter", "clarify", "risky_action", "approval", "finalize"):
        assert node in diagram, f"Node '{node}' missing from diagram"
```

---

## Section 2: Time-Travel Replay

### Implementation

**`scripts/demo_time_travel.py`** (new):
- Use `build_checkpointer("sqlite", "outputs/time_travel.db")`
- Build graph, run S02 (`"Please lookup order status for order 12345"`) with `thread_id="thread-time-travel"`
- Call `graph.get_state_history({"configurable": {"thread_id": "thread-time-travel"}})`
- Iterate snapshots (from newest to oldest), print each: step index, `route`, `len(events)`, `attempt`
- Assert earliest snapshot has `route=""` (initial state before classify)

### Tests

**`tests/test_time_travel.py`** (new):

```python
def test_state_history_has_multiple_snapshots():
    from langgraph_agent_lab.graph import build_graph
    from langgraph_agent_lab.persistence import build_checkpointer
    from langgraph_agent_lab.state import Route, Scenario, initial_state
    checkpointer = build_checkpointer("sqlite", "outputs/test_time_travel.db")
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(id="tt-test", query="Please lookup order status for order 99", expected_route=Route.TOOL)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    graph.invoke(state, config=config)
    snapshots = list(graph.get_state_history(config))
    assert len(snapshots) >= 3

def test_state_history_earliest_has_empty_route():
    from langgraph_agent_lab.graph import build_graph
    from langgraph_agent_lab.persistence import build_checkpointer
    from langgraph_agent_lab.state import Route, Scenario, initial_state
    checkpointer = build_checkpointer("sqlite", "outputs/test_time_travel2.db")
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(id="tt-test2", query="Please lookup order status for order 88", expected_route=Route.TOOL)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    graph.invoke(state, config=config)
    snapshots = list(graph.get_state_history(config))
    earliest = snapshots[-1]
    assert earliest.values.get("route", "") == ""
```

---

## Section 3: LangSmith Tracing

### Implementation

**`pyproject.toml`**: add `langsmith` to `[project.dependencies]`

**`.env.example`**: add:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_key_here
LANGCHAIN_PROJECT=day08-lab
```

No code changes needed. LangGraph reads these env vars automatically and sends traces to LangSmith Studio when set.

### Tests

None — env-var-only integration. Verified by setting vars and running `make run-scenarios`, then checking LangSmith Studio UI.

---

## Files Changed

| File | Action |
|------|--------|
| `tests/test_diagram.py` | Create — 2 tests |
| `tests/test_time_travel.py` | Create — 2 tests |
| `scripts/generate_diagram.py` | Create — diagram export script |
| `scripts/demo_time_travel.py` | Create — time-travel demo script |
| `src/langgraph_agent_lab/report.py` | Modify — append graph diagram section |
| `docs/graph.md` | Create — committed Mermaid diagram output |
| `pyproject.toml` | Modify — add langsmith dependency |
| `.env.example` | Modify — add LangSmith env vars |

---

## Implementation Order

1. **Graph diagram** — `tests/test_diagram.py` (red) → `scripts/generate_diagram.py` → `report.py` update → green → commit
2. **Time-travel replay** — `tests/test_time_travel.py` (red) → `scripts/demo_time_travel.py` → green → commit
3. **LangSmith tracing** — `uv add langsmith` → `.env.example` → `pyproject.toml` → commit

---

## Final Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -v      # 41 passed
uv run python scripts/generate_diagram.py                      # writes docs/graph.md
uv run python scripts/demo_time_travel.py                      # prints snapshot sequence
make run-scenarios && make grade-local                         # success_rate=100.00%
```
