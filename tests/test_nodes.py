# tests/test_nodes.py
from langgraph_agent_lab.nodes import (
    classify_node,
    dead_letter_node,
    evaluate_node,
    intake_node,
)


# --- classify_node ---

def test_classify_simple():
    state = {"query": "How do I reset my password?"}
    result = classify_node(state)
    assert result["route"] == "simple"


def test_classify_tool_check():
    assert classify_node({"query": "check my account balance"})["route"] == "tool"


def test_classify_tool_track():
    assert classify_node({"query": "track my shipment"})["route"] == "tool"


def test_classify_tool_find():
    assert classify_node({"query": "find order 123"})["route"] == "tool"


def test_classify_tool_search():
    assert classify_node({"query": "search for a product"})["route"] == "tool"


def test_classify_risky_cancel():
    assert classify_node({"query": "cancel my order please"})["route"] == "risky"


def test_classify_risky_remove():
    assert classify_node({"query": "remove my account"})["route"] == "risky"


def test_classify_risky_revoke():
    assert classify_node({"query": "revoke access token"})["route"] == "risky"


def test_classify_error_error():
    assert classify_node({"query": "system error occurred"})["route"] == "error"


def test_classify_error_crash():
    assert classify_node({"query": "server crash detected"})["route"] == "error"


def test_classify_error_unavailable():
    assert classify_node({"query": "service unavailable"})["route"] == "error"


def test_classify_missing_info():
    assert classify_node({"query": "Can you fix it?"})["route"] == "missing_info"


def test_classify_priority_risky_over_tool():
    # "cancel" (risky) beats "lookup" (tool)
    assert classify_node({"query": "cancel and lookup order"})["route"] == "risky"


def test_classify_priority_risky_over_error():
    # "delete" (risky) beats "timeout" (error)
    assert classify_node({"query": "delete timeout record"})["route"] == "risky"


# --- intake_node ---

def test_intake_strips_whitespace():
    result = intake_node({"query": "  hello world  "})
    assert result["query"] == "hello world"


# --- evaluate_node ---

def test_evaluate_success():
    state = {"tool_results": ["mock-tool-result for scenario=x"]}
    result = evaluate_node(state)
    assert result["evaluation_result"] == "success"


def test_evaluate_needs_retry():
    state = {"tool_results": ["ERROR: transient failure attempt=0"]}
    result = evaluate_node(state)
    assert result["evaluation_result"] == "needs_retry"


# --- dead_letter_node ---

def test_dead_letter_sets_answer():
    result = dead_letter_node({"attempt": 3})
    assert "manual review" in result["final_answer"].lower()
