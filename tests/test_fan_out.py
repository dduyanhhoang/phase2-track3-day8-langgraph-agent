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
