import logging
from src.state import AgentState
from langchain_core.runnables import RunnableConfig

# --- MOCK DATABASE (SIMULATION) ---
# In a real production app, this would be a Qdrant 'scroll()' call.
USER_PROFILES = {
    "officer_keshav": (
        "ROLE: Presiding Officer\n"
        "LOCATION: Rural Constituency (Hilly Terrain)\n"
        "HISTORY: Previously asked about VVPAT battery replacement.\n"
        "PREFERENCE: Needs exact form numbers and official protocols."
    ),
    "journalist_riya": (
        "ROLE: Investigative Journalist\n"
        "LOCATION: New Delhi\n"
        "HISTORY: Investigating EVM hacking myths.\n"
        "PREFERENCE: Needs technical specs, source citations, and 'trust_score' analysis."
    ),
    "citizen_raj": (
        "ROLE: First-time Voter\n"
        "LOCATION: Mumbai\n"
        "HISTORY: None.\n"
        "PREFERENCE: Needs simple explanations in Hindi/English mix."
    )
}

def load_memory_node(state: AgentState, config: RunnableConfig):
    """
    Node 0: Context Loader
    Fetches the User's Long-Term Profile based on their User ID.
    """
    # 1. Get User ID from the Config (passed from main.py)
    # Default to 'guest' if not provided
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", "guest")
    
    print(f"\n📂 [LOADER] Fetching Long-Term Memory for: '{user_id}'")
    
    # 2. Retrieve Profile
    # Fallback for unknown users
    context = USER_PROFILES.get(user_id, "ROLE: General Public\nPREFERENCE: Standard info.")
    
    # 3. (Optional) Real Qdrant Logic would go here:
    # results = client.scroll(collection_name="user_profiles", ...)
    
    print(f"   -> Found Context: {context[:50]}...")
    
    # 4. Inject into State
    return {"user_context": context}