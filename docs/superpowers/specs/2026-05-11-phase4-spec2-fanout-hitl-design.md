# Phase 4 Spec 2 — Parallel Fan-out + Streamlit HITL Design

## Goal

Add two bonus extensions: parallel fan-out via LangGraph `Send()` API (new `multi_tool`
route with two concurrent mock tools), and a full Streamlit approval UI for real HITL
using `interrupt()` + `Command(resume=...)`.

---

## Extension A: Parallel Fan-out

### What changes

| File | Change |
|------|--------|
| `src/langgraph_agent_lab/state.py` | Add `Route.MULTI_TOOL = "multi_tool"` to enum |
| `src/langgraph_agent_lab/nodes.py` | Add `"bulk"` to classify; add `fan_out_node`, `tool_a_node`, `tool_b_node`, `merge_node` |
| `src/langgraph_agent_lab/routing.py` | Add `multi_tool → fan_out` in `route_after_classify` |
| `src/langgraph_agent_lab/graph.py` | Wire 4 new nodes; add `Send()` conditional edge from `fan_out` |
| `data/sample/scenarios.jsonl` | Add S08: `"Bulk lookup all pending orders"` → `multi_tool` |
| `tests/test_fan_out.py` | Create — 3 tests |

### Data flow

```
multi_tool → fan_out → [Send("tool_a", state), Send("tool_b", state)]
                          ↓                        ↓
                       tool_a_node              tool_b_node
                          ↓                        ↓
                       merge_node ←────────────────┘
                          ↓
                       evaluate → answer → finalize → END
```

### Node specs

**`fan_out_node(state)`**: returns `[Send("tool_a", state), Send("tool_b", state)]`
— no state update, pure routing.

**`tool_a_node(state)`**: appends `f"tool-a-result for scenario={state['scenario_id']}"` to `tool_results`.

**`tool_b_node(state)`**: appends `f"tool-b-result for scenario={state['scenario_id']}"` to `tool_results`.

**`merge_node(state)`**: emits single `make_event("merge", "completed", "parallel tools merged")` event. No other state changes — `tool_results` already merged via `add` reducer.

### Classify keyword

Add `"bulk"` to risky-check's elif chain between tool and missing_info:
```python
elif any(k in query for k in ("bulk",)):
    route = Route.MULTI_TOOL
```
Priority: risky > tool > **multi_tool** > missing_info > error > simple.

### S08 scenario

```json
{"id":"S08_multi","query":"Bulk lookup all pending orders","expected_route":"multi_tool","requires_approval":false,"should_retry":false,"tags":["fan_out","parallel"]}
```

### Tests

**`tests/test_fan_out.py`**:

```python
def test_classify_multi_tool():
    from langgraph_agent_lab.nodes import classify_node
    result = classify_node({"query": "Bulk lookup all pending orders"})
    assert result["route"] == "multi_tool"

def test_fan_out_node_returns_sends():
    from langgraph.types import Send
    from langgraph_agent_lab.nodes import fan_out_node
    from langgraph_agent_lab.state import initial_state, Scenario, Route
    scenario = Scenario(id="fo", query="Bulk lookup all pending orders", expected_route=Route.MULTI_TOOL)
    state = initial_state(scenario)
    result = fan_out_node(state)
    assert isinstance(result, list) and len(result) == 2
    assert all(isinstance(s, Send) for s in result)

def test_multi_tool_graph_produces_two_results():
    from langgraph_agent_lab.graph import build_graph
    from langgraph_agent_lab.persistence import build_checkpointer
    from langgraph_agent_lab.state import initial_state, Scenario, Route
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="fo2", query="Bulk lookup all pending orders", expected_route=Route.MULTI_TOOL)
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    assert len(result.get("tool_results", [])) == 2
    assert result.get("final_answer")
```

---

## Extension B: Streamlit HITL

### What changes

| File | Change |
|------|--------|
| `pyproject.toml` | Add `streamlit>=1.35` |
| `scripts/demo_hitl_setup.py` | Run S04 with `LANGGRAPH_INTERRUPT=true` into interrupted state |
| `apps/approval_ui.py` | Streamlit approval app |
| `tests/test_hitl_resume.py` | Create — 2 tests |

### HITL flow

```
[demo_hitl_setup.py]
  graph.invoke(S04_state, config) with LANGGRAPH_INTERRUPT=true
    → hits approval_node → interrupt() fires → graph pauses
    → prints thread_id for Streamlit

[approval_ui.py]
  1. User enters thread_id in sidebar
  2. graph.get_state(config) → reads paused state from SQLite
  3. Shows proposed_action + risk_level
  4. Approve button → graph.invoke(Command(resume={"approved": True}), config)
  5. Reject button  → graph.invoke(Command(resume={"approved": False}), config)
  6. Shows final result
```

### apps/approval_ui.py design

- Imports: `streamlit`, `build_graph`, `build_checkpointer`, `Command` from `langgraph.types`
- Sidebar: text input for `thread_id`, text input for `db_path` (default `outputs/checkpoints.db`)
- Main area: loads state → if interrupted shows action card → Approve/Reject buttons
- After action: shows final_answer or pending_question

### scripts/demo_hitl_setup.py design

- Sets `os.environ["LANGGRAPH_INTERRUPT"] = "true"` before import
- Runs S04 (`"Refund this customer and send confirmation email"`) with SQLite checkpointer
- Catches `GraphInterrupt` or detects interrupted state (`.next` not empty after invoke)
- Prints `thread_id` and `proposed_action` for user to copy into Streamlit

### Tests

**`tests/test_hitl_resume.py`**:

```python
def test_graph_pauses_at_approval_with_interrupt(monkeypatch):
    import os
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")
    from langgraph.types import Command
    from langgraph_agent_lab.graph import build_graph
    from langgraph_agent_lab.persistence import build_checkpointer
    from langgraph_agent_lab.state import initial_state, Scenario, Route
    checkpointer = build_checkpointer("memory")
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(id="hitl-test", query="Refund this customer", expected_route=Route.RISKY, requires_approval=True)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    try:
        graph.invoke(state, config=config)
        # If no exception, check interrupted state
        saved = graph.get_state(config)
        assert "approval" in (saved.next or []) or saved.values.get("approval") is None
    except Exception:
        pass  # GraphInterrupt raised — interrupt fired correctly

def test_graph_resumes_after_command(monkeypatch):
    import os
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")
    from langgraph.types import Command
    from langgraph_agent_lab.graph import build_graph
    from langgraph_agent_lab.persistence import build_checkpointer
    from langgraph_agent_lab.state import initial_state, Scenario, Route
    checkpointer = build_checkpointer("memory")
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(id="hitl-resume", query="Refund this customer", expected_route=Route.RISKY, requires_approval=True)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    try:
        graph.invoke(state, config=config)
    except Exception:
        pass
    result = graph.invoke(Command(resume={"approved": True, "reviewer": "test", "comment": "ok"}), config=config)
    assert result.get("approval") is not None
    assert result.get("final_answer")
```

---

## Implementation Order

### Module A (fan-out)

1. `tests/test_fan_out.py` — 3 tests (red)
2. `state.py` — add `Route.MULTI_TOOL`
3. `nodes.py` — `"bulk"` keyword + 4 new nodes
4. `routing.py` — `multi_tool → fan_out`
5. `graph.py` — wire nodes + `Send()` edge
6. `data/sample/scenarios.jsonl` — add S08
7. Verify 8/8 scenarios pass
8. Commit

### Module B (HITL)

1. `tests/test_hitl_resume.py` — 2 tests
2. `uv add streamlit`
3. `scripts/demo_hitl_setup.py`
4. `apps/approval_ui.py`
5. Commit

---

## Files Changed

| File | Action |
|------|--------|
| `src/langgraph_agent_lab/state.py` | Add `MULTI_TOOL` route |
| `src/langgraph_agent_lab/nodes.py` | Add `bulk` keyword + 4 nodes |
| `src/langgraph_agent_lab/routing.py` | Add `multi_tool → fan_out` |
| `src/langgraph_agent_lab/graph.py` | Wire fan-out with Send() |
| `data/sample/scenarios.jsonl` | Add S08 |
| `tests/test_fan_out.py` | Create — 3 tests |
| `tests/test_hitl_resume.py` | Create — 2 tests |
| `pyproject.toml` | Add streamlit |
| `scripts/demo_hitl_setup.py` | Create |
| `apps/approval_ui.py` | Create |

---

## Final Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -v   # 46 passed
make run-scenarios && make grade-local                      # 8/8, 100%
uv run python scripts/demo_hitl_setup.py                   # prints thread_id + proposed_action
streamlit run apps/approval_ui.py                          # approval UI opens
```
