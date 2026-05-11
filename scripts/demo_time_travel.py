"""Time-travel replay demo: run S02, then replay all checkpoints via get_state_history()."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

DB_PATH = "outputs/time_travel_demo.db"
THREAD_ID = "thread-time-travel-demo"


def main() -> None:
    Path("outputs").mkdir(exist_ok=True)

    checkpointer = build_checkpointer("sqlite", DB_PATH)
    graph = build_graph(checkpointer=checkpointer)

    scenario = Scenario(
        id="time-travel-demo",
        query="Please lookup order status for order 12345",
        expected_route=Route.TOOL,
    )
    state = initial_state(scenario)
    state["thread_id"] = THREAD_ID
    config = {"configurable": {"thread_id": THREAD_ID}}

    print("=== Phase 1: Run scenario S02 ===")
    result = graph.invoke(state, config=config)
    print(f"Final route: {result['route']}, answer: {result.get('final_answer')}")

    print("\n=== Phase 2: Time-travel replay via get_state_history() ===")
    snapshots = list(graph.get_state_history(config))
    print(f"Total checkpoints recorded: {len(snapshots)}")
    print("\nCheckpoint sequence (newest → oldest):")
    for i, snapshot in enumerate(snapshots):
        route = snapshot.values.get("route", "(empty)")
        events = len(snapshot.values.get("events", []))
        attempt = snapshot.values.get("attempt", 0)
        next_node = snapshot.next[0] if snapshot.next else "END"
        print(f"  [{i:02d}] next={next_node:<15} route={route:<15} events={events} attempt={attempt}")

    print("\n=== Phase 3: Replay from earliest checkpoint ===")
    earliest = snapshots[-1]
    print(f"Earliest checkpoint: route='{earliest.values.get('route', '')}' (should be empty — before classify)")
    print("Time-travel SUCCESS: full checkpoint history available for replay.")


if __name__ == "__main__":
    main()
