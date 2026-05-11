"""Crash-resume demo: run a scenario, simulate crash, resume from checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

DB_PATH = "outputs/demo_checkpoints.db"
THREAD_ID = "thread-crash-demo"


def run_first_invocation() -> None:
    """Run one scenario and store state to SQLite."""
    checkpointer = build_checkpointer("sqlite", DB_PATH)
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(
        id="crash-demo",
        query="Please lookup order status for order 99999",
        expected_route=Route.TOOL,
    )
    state = initial_state(scenario)
    state["thread_id"] = THREAD_ID
    result = graph.invoke(state, config={"configurable": {"thread_id": THREAD_ID}})
    print(f"[first run] route={result['route']} answer={result.get('final_answer')}")
    print(f"[first run] events={len(result.get('events', []))} — state persisted to {DB_PATH}")


def run_resume() -> None:
    """Resume from checkpoint — no initial state needed."""
    checkpointer = build_checkpointer("sqlite", DB_PATH)
    graph = build_graph(checkpointer=checkpointer)
    saved = graph.get_state({"configurable": {"thread_id": THREAD_ID}})
    if saved is None or saved.values is None:
        print("[resume] ERROR: no checkpoint found — run first invocation first")
        sys.exit(1)
    restored_route = saved.values.get("route")
    restored_answer = saved.values.get("final_answer")
    print(f"[resume] restored route={restored_route} answer={restored_answer}")
    print("[resume] crash-resume SUCCESS: state survived simulated crash")


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    print("=== Phase 1: first invocation ===")
    run_first_invocation()
    print("\n=== Phase 2: simulated crash — process would have died here ===")
    print("=== Phase 3: resume from SQLite checkpoint ===")
    run_resume()
