# apps/approval_ui.py
"""Streamlit HITL approval UI for LangGraph risky action review."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer

st.set_page_config(page_title="HITL Approval UI", page_icon="⚠️", layout="centered")
st.title("⚠️ HITL Approval UI")
st.caption("LangGraph human-in-the-loop approval for risky actions")

with st.sidebar:
    st.header("Connection")
    db_path = st.text_input("DB path", value="outputs/hitl_demo.db")
    thread_id = st.text_input("Thread ID", value="thread-hitl-demo")
    load = st.button("Load state")

if load or (db_path and thread_id):
    try:
        checkpointer = build_checkpointer("sqlite", db_path)
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        saved = graph.get_state(config)

        if saved is None or not saved.values:
            st.warning("No state found for this thread_id. Run demo_hitl_setup.py first.")
        elif not saved.next:
            st.success("Graph already completed.")
            st.json({
                "route": saved.values.get("route"),
                "final_answer": saved.values.get("final_answer"),
                "approval": saved.values.get("approval"),
            })
        elif "approval" in list(saved.next):
            proposed = saved.values.get("proposed_action", "Unknown action")
            risk = saved.values.get("risk_level", "unknown")
            query = saved.values.get("query", "")

            st.error(f"⚠️ Pending approval — risk level: **{risk.upper()}**")
            st.markdown(f"**Query:** {query}")
            st.markdown(f"**Proposed action:** {proposed}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", type="primary", use_container_width=True):
                    result = graph.invoke(
                        Command(resume={"approved": True, "reviewer": "streamlit-user", "comment": "approved via UI"}),
                        config=config,
                    )
                    st.success("Approved! Graph resumed.")
                    st.json({
                        "final_answer": result.get("final_answer"),
                        "approval": result.get("approval"),
                    })
            with col2:
                if st.button("❌ Reject", type="secondary", use_container_width=True):
                    result = graph.invoke(
                        Command(resume={"approved": False, "reviewer": "streamlit-user", "comment": "rejected via UI"}),
                        config=config,
                    )
                    st.warning("Rejected. Graph resumed with clarify path.")
                    st.json({
                        "final_answer": result.get("final_answer"),
                        "pending_question": result.get("pending_question"),
                    })
        else:
            st.info(f"Graph paused at unexpected node: {saved.next}")
    except Exception as exc:
        st.error(f"Error: {exc}")
