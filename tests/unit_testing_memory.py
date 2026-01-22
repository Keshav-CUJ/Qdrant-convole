import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (mas_election_agent/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to Python's search path
sys.path.append(parent_dir)

from src.nodes.loader import load_memory_node
from src.nodes.memory import memory_update_node
from src.state import AgentState

# Define a Test User
TEST_USER_ID = "officer_rahul_09"
MOCK_CONFIG = {"configurable": {"user_id": TEST_USER_ID}}

def test_memory_write():
    print(f"\n✍️ TEST 1: Writing Memory for '{TEST_USER_ID}'")
    print("-" * 50)
    
    # 1. Simulate a Conversation where user reveals info
    # User says: "I am a Presiding Officer in Mumbai. I want technical answers."
    
    mock_state = {
        "messages": [
            type("obj", (object,), {"content": "Hi, I am Rahul, a Presiding Officer posted in Mumbai."}),
            type("obj", (object,), {"content": "Understood. How can I help?"}),
            type("obj", (object,), {"content": "I need technical details about Form 17C. Don't give me simple answers."}),
            type("obj", (object,), {"content": "Noted. I will provide technical documentation."})
        ]
    }
    
    # 2. Run the Memory Update Node
    print("🧠 analyzing conversation with LLM...")
    try:
        memory_update_node(mock_state, MOCK_CONFIG)
        print("✅ Memory Update Node finished without error.")
    except Exception as e:
        print(f"❌ Error in Memory Writer: {e}")

def test_memory_read():
    print(f"\n📖 TEST 2: Reading Memory for '{TEST_USER_ID}'")
    print("-" * 50)
    
    # 1. Run the Loader Node
    # It should fetch the profile we just saved
    result = load_memory_node({}, MOCK_CONFIG)
    context = result.get("user_context", "")
    
    print("\n⬇️ RETRIEVED CONTEXT ⬇️")
    print(context)
    print("⬆️ ----------------- ⬆️")
    
    # 2. Verify Logic
    if "Official" in context or "Officer" in context:
        print("🎉 SUCCESS: It remembered the Persona (Officer)!")
    else:
        print("⚠️ WARNING: Persona missing.")
        
    if "Mumbai" in context:
        print("🎉 SUCCESS: It remembered the Location (Mumbai)!")
    
    if "Technical" in context or "Detailed" in context:
        print("🎉 SUCCESS: It remembered the Interaction Style!")

if __name__ == "__main__":
    # Run Write first, then Read
    test_memory_write()
    
    print("\n⏳ Waiting 2 seconds for DB to index...")
    time.sleep(2) 
    
    test_memory_read()