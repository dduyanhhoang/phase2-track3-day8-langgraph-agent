# tests/test_time_travel.py
import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("langgraph") is None,
    reason="langgraph not installed",
)

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


def _run_s02(db_path: str):
    checkpointer = build_checkpointer("sqlite", db_path)
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(
        id="tt-s02",
        query="Please lookup order status for order 99",
        expected_route=Route.TOOL,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    graph.invoke(state, config=config)
    return graph, config


def test_state_history_has_multiple_snapshots(tmp_path):
    db = str(tmp_path / "tt1.db")
    graph, config = _run_s02(db)
    snapshots = list(graph.get_state_history(config))
    assert len(snapshots) >= 3


def test_state_history_earliest_has_empty_route(tmp_path):
    db = str(tmp_path / "tt2.db")
    graph, config = _run_s02(db)
    snapshots = list(graph.get_state_history(config))
    earliest = snapshots[-1]
    assert earliest.values.get("route", "") == ""
