import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (mas_election_agent/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to Python's search path
sys.path.append(parent_dir)

import json
from src.state import AgentState
from src.nodes.loader import load_memory_node
from src.nodes.researcher import query_gen_node, search_execution_node, responder_node

# --- CONFIG FOR TESTS ---
MOCK_CONFIG = {"configurable": {"user_id": "officer_keshav"}}



def test_query_gen_node():
    print("\n🧪 TEST 2: Query Generator Node (Decision Making)")
    print("-" * 40)
    
    # 1. Simulate State (Input from User)
    state = {
        "messages": [type("obj", (object,), {"content": "Is this machine hacked?"})], # Mock Message
        "user_context": "User is a skeptical citizen.",
        "current_image_path": "assets/test_evm.jpg" # SIMULATE IMAGE UPLOAD
    }
    
    # 2. Run Node
    result = query_gen_node(state)
    plans = result.get("search_plans")
    
    # 3. Verify
    print(f"✅ Generated Plans: {json.dumps(plans, indent=2)}")
    
    # Critical Logic Checks
    assert isinstance(plans, list), "Output must be a LIST of plans"
    assert len(plans) >= 1, "Must generate at least one plan"
    
    # Check if it correctly decided to use Image Search
    tools_chosen = [p['tool'] for p in plans]
    print(f"✅ Tools Selected: {tools_chosen}")
    if "search_image" in tools_chosen:
        print("🎉 SUCCESS: Agent correctly detected image input.")
    else:
        print("⚠️ WARNING: Agent ignored the image input.")

def test_search_exec_node():
    print("\n🧪 TEST 3: Search Executor Node (Parsing Plans)")
    print("-" * 40)
    
    # 1. Simulate State (Output from Query Gen)
    state = {
        "search_plans": [
            {
                "tool": "search_hybrid", 
                "query": "EVM hacking myths", 
                "filters": {},
                "purpose": "Fact check"
            }
        ]
    }
    
    # 2. Run Node
    # Note: This will actually call Qdrant. Ensure your DB is running.
    result = search_execution_node(state)
    docs = result.get("retrieved_docs")
    
    # 3. Verify
    print(f"✅ Retrieved Docs Length: {len(docs)} chars")
    print(f"✅ Preview:\n{docs[:200]}...")
    assert "=== STEP 1" in docs, "Output formatting is missing step headers"

def test_responder_node():
    print("\n🧪 TEST 4: Responder Node (The Verdict)")
    print("-" * 40)
    
    # 1. Simulate State (Rich Evidence)
    mock_docs = """
    --- EVIDENCE START ---
    TITLE: Fact Check #123
    STATUS: official_truth
    TEXT: Claims about Bluetooth EVMs are FALSE. EVMs are standalone.
    SOURCE URL: https://eci.gov.in/factcheck
    INTENT: Spread_correct_info
    --- EVIDENCE END ---
    """
    
    state = {
        "messages": [type("obj", (object,), {"content": "I heard EVMs have bluetooth?"})],
        "retrieved_docs": mock_docs,
        "user_context": "User is a voter."
    }
    
    # 2. Run Node
    result = responder_node(state)
    final_msg = result["messages"][0].content
    
    # 3. Verify
    print(f"✅ Final Answer:\n{final_msg}")
    
    # Check if it followed instructions
    if "VERIFIED" in final_msg or "MISINFORMATION" in final_msg:
        print("🎉 SUCCESS: Verdict included.")
    else:
        print("⚠️ WARNING: Verdict missing.")
        
    if "https://" in final_msg:
        print("🎉 SUCCESS: URL cited.")
    else:
        print("⚠️ WARNING: Source URL not cited.")

if __name__ == "__main__":
    # Uncomment the one you want to debug, or run all
    
    # test_query_gen_node()
    # test_search_exec_node()
    # test_responder_node()