# src/graph/workflow.py
from langgraph.graph import StateGraph, START, END
from src.state import AgentState

# Import your nodes
from src.nodes.loader import load_memory_node   # (The Hippocampus)
from src.nodes.researcher import (              # (The Researcher Parts)
    query_gen_node, 
    search_execution_node, 
    responder_node
)

def build_graph():
    # 1. Initialize Graph
    workflow = StateGraph(AgentState)

    # 2. Add Nodes
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("generate_query", query_gen_node)
    workflow.add_node("execute_search", search_execution_node)
    workflow.add_node("write_answer", responder_node)

    # 3. Define Edges (The Flow)
    # Start -> Load Memory -> Gen Query -> Search -> Write Answer -> End
    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "generate_query")
    workflow.add_edge("generate_query", "execute_search")
    workflow.add_edge("execute_search", "write_answer")
    workflow.add_edge("write_answer", END)

    # 4. Compile
    return workflow.compile()