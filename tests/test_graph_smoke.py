import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed in local environment")

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("How do I reset my password?", Route.SIMPLE.value),
        ("Please lookup order status for order 123", Route.TOOL.value),
        ("Refund this customer", Route.RISKY.value),
    ],
)
def test_graph_runs_basic_routes(query, expected_route):
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="smoke", query=query, expected_route=Route(expected_route))
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    assert result["route"] == expected_route
    assert result.get("final_answer") or result.get("pending_question")


def test_missing_info_path():
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="smoke-missing", query="Can you fix it?", expected_route=Route.MISSING_INFO)
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    assert result["route"] == Route.MISSING_INFO.value
    assert result.get("pending_question")


def test_error_retry_path():
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(
        id="smoke-error",
        query="Timeout failure while processing request",
        expected_route=Route.ERROR,
        should_retry=True,
        max_attempts=3,
    )
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    assert result["route"] == Route.ERROR.value
    assert result.get("final_answer")
    retry_count = sum(1 for e in result.get("events", []) if e.get("node") == "retry")
    assert retry_count >= 1


def test_dead_letter_path():
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(
        id="smoke-dead",
        query="Timeout failure while processing request",
        expected_route=Route.ERROR,
        should_retry=True,
        max_attempts=1,
    )
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    assert "manual review" in (result.get("final_answer") or "").lower()


def test_risky_approval_path():
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(
        id="smoke-risky",
        query="Refund this customer and send confirmation",
        expected_route=Route.RISKY,
        requires_approval=True,
    )
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    assert result["route"] == Route.RISKY.value
    assert result.get("approval") is not None
    assert result.get("final_answer")
