# src/state.py
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Session History (The Chat)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Long-Term Memory (Context)
    user_context: str
    
    # Workflow Data (Passing data between nodes)
    search_plans: List[dict]      # Output of Query Generator
    retrieved_docs: str    # Output of Search Tool
    current_image_path: Optional[str] = None