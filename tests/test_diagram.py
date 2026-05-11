# tests/test_diagram.py
import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("langgraph") is None,
    reason="langgraph not installed",
)

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer


def _diagram() -> str:
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    return graph.get_graph().draw_mermaid()


def test_draw_mermaid_returns_string():
    diagram = _diagram()
    assert isinstance(diagram, str) and len(diagram) > 0
    assert "flowchart" in diagram or "graph" in diagram.lower()


def test_diagram_contains_all_nodes():
    diagram = _diagram()
    for node in (
        "intake", "classify", "answer", "tool", "evaluate",
        "retry", "dead_letter", "clarify", "risky_action", "approval", "finalize",
    ):
        assert node in diagram, f"Node '{node}' missing from Mermaid diagram"
