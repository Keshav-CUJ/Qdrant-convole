import uuid
import os
from src.graph.workflow import build_graph

# 1. Setup
app = build_graph()
current_user = "officer_keshav"

# Function to start a fresh session
def create_new_session():
    return str(uuid.uuid4())

# 2. Start First Chat
current_thread_id = create_new_session()
print(f"🔵 Starting Chat Session 1 (ID: {current_thread_id})")
print("💡 TIP: To upload an image, type: 'Your Question | path/to/image.jpg'")
print("   Example: 'Is this real? | assets/sample_evm.jpg'")

# Chat Loop
while True:
    raw_input = "ECI is being tagged on many occasions for matters related to conduct of  local body elections in various states/UTs"
    
    # --- LOGIC TO HANDLE NEW CHAT ---
    if raw_input.lower() == "new":
        current_thread_id = create_new_session()
        print(f"\n✨ NEW CHAT STARTED (ID: {current_thread_id})")
        continue
    
    if raw_input.lower() in ["quit", "exit"]: break

    # --- PARSE INPUT FOR IMAGE ---
    user_text = raw_input
    image_path = None
    
    if "|" in raw_input:
        parts = raw_input.split("|")
        user_text = parts[0].strip()
        image_path = parts[1].strip()
        
        # Verify file exists to avoid unnecessary errors
        # if not os.path.exists(image_path):
        #     print(f"⚠️ Warning: Image file '{image_path}' not found. Sending text only.")
        #     image_path = None
        # else:
        #     print(f"📎 Image Attached: {image_path}")

    # --- PREPARE INPUT STATE ---
    # This matches your AgentState definition in src/state.py
    input_payload = {
        "messages": [("user", user_text)],
        "current_image_path": image_path  # <--- PASSING THE IMAGE HERE
    }

    # Run Graph with CURRENT thread_id
    config = {
        "configurable": {
            "user_id": current_user,
            "thread_id": current_thread_id
        }
    }
    
    print("🤖 Agent Thinking...")
    try:
        for event in app.stream(input_payload, config=config):
            for k, v in event.items():
                # We print the output of specific nodes to see progress
                if k == "generate_query":
                    print(f"   ↳ 🧠 Plan: {len(v.get('search_plans', []))} searches generated.")
                elif k == "write_answer":
                    print(f"\nAgent: {v['messages'][-1].content}")
    except Exception as e:
        print(f"❌ Error: {e}")
    break