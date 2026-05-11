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
