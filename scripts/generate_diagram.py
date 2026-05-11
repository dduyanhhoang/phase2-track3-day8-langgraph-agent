# scripts/generate_diagram.py
"""Generate Mermaid diagram from LangGraph workflow and write to docs/graph.md."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer

OUTPUT = Path(__file__).parent.parent / "docs" / "graph.md"


def main() -> None:
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    diagram = graph.get_graph().draw_mermaid()
    content = f"# Agent Graph Diagram\n\n```mermaid\n{diagram}\n```\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Diagram written to {OUTPUT}")
    print(diagram)


if __name__ == "__main__":
    main()
