import logging
from src.state import AgentState
from langchain_core.runnables import RunnableConfig
from qdrant_client import models
from src.config import client, MEMORY_COLLECTION_NAME
from src.state import AgentState
import uuid
# # --- MOCK DATABASE (SIMULATION) ---
# # In a real production app, this would be a Qdrant 'scroll()' call.
# USER_PROFILES = {
#     "officer_keshav": (
#         "ROLE: Presiding Officer\n"
#         "LOCATION: Rural Constituency (Hilly Terrain)\n"
#         "HISTORY: Previously asked about VVPAT battery replacement.\n"
#         "PREFERENCE: Needs exact form numbers and official protocols."
#     ),
#     "journalist_riya": (
#         "ROLE: Investigative Journalist\n"
#         "LOCATION: New Delhi\n"
#         "HISTORY: Investigating EVM hacking myths.\n"
#         "PREFERENCE: Needs technical specs, source citations, and 'trust_score' analysis."
#     ),
#     "citizen_raj": (
#         "ROLE: First-time Voter\n"
#         "LOCATION: Mumbai\n"
#         "HISTORY: None.\n"
#         "PREFERENCE: Needs simple explanations in Hindi/English mix."
#     )
# }

COLLECTION_NAME=MEMORY_COLLECTION_NAME

def get_user_uuid(user_id: str) -> str:
    """
    Generates a DETERMINISTIC UUID from a user_id.
    'officer_keshav' -> always returns '36f1c4...9a'
    """
    NAMESPACE_OID = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8") # Standard namespace
    return str(uuid.uuid5(NAMESPACE_OID, user_id))


def load_memory_node(state: AgentState, config: RunnableConfig):
    # 1. Calculate the ID again (It's deterministic!)
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", "guest")
    point_id = get_user_uuid(user_id) 
    
    print(f"📂 [LOADER] Fetching Profile for User: {user_id} (ID: {point_id})")
    
    # 2. DIRECT RETRIEVE (Not Search)
    # We ask Qdrant: "Give me the record with ID = X"
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
        with_vectors=False # We don't need the vector numbers, just the text
    )
    
    # 3. Handle Result
    if points:
        # Found them!
        payload = points[0].payload
        
        # Format it for the Agent to read
        context_str = (
            f"NAME: {payload.get('name', 'Unknown')}\n"
            f"LOCATION: {payload.get('location', 'Unknown')}\n"
            f"PERSONA: {payload.get('persona', 'Unknown')}\n"
            f"STYLE: {payload.get('interaction_style', 'Normal')}\n"
            f"PREFERENCES: {payload.get('content_preferences', {})}\n"
            f"SUMMARY: {payload.get('summary', '')}"
        )
        print("   -> ✅ Found existing profile.")
    else:
        # New User
        context_str = "PERSONA: New User\nSTYLE: Helpful & Clear"
        print("   -> 🆕 New user created.")

    return {"user_context": context_str}