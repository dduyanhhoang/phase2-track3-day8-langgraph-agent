# scripts/demo_hitl_setup.py
"""Run a risky scenario into an interrupted state for the Streamlit HITL demo.

Usage:
    uv run python scripts/demo_hitl_setup.py

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
