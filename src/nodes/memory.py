import json
from langchain_core.messages import SystemMessage
from qdrant_client import models
from src.state import AgentState
from src.config import client, MEMORY_COLLECTION_NAME 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
import uuid
import os
from dotenv import load_dotenv
from src.config import dense_text_model
load_dotenv()



def get_user_uuid(user_id: str) -> str:
    """
    Generates a DETERMINISTIC UUID from a user_id.
    'officer_keshav' -> always returns '36f1c4...9a'
    """
    NAMESPACE_OID = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8") # Standard namespace
    return str(uuid.uuid5(NAMESPACE_OID, user_id))


# --- LLM for Summarization ---
llm_memory = llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY2"),
    temperature=0,
    max_retries=2,
    # Safety settings to prevent blocking legitimate election queries
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
)

MEMORY_PROMPT = """You are an agent which handles ***long term memory*** of the user in Qdrant collection for Misinformation Detection System.
Your role is to build high-fidelity psychological profile of the user based on their interaction,which can be used to personalized user interactions for misinformation detection.

### INPUT DATA
CURRENT PROFILE:
{current_profile}
NEW INTERACTION:
User: {last_user_msg}
Agent: {last_agent_msg}

TASK:
1. Update the user's summary, persona, interaction style, location, name and content preferences if changed.
2. Add new important facts (e.g., "User hates technical jargon").
3. IGNORE transient info (e.g., "What time is it?").

### EXTRACTION RULES

1. **Identify Persona:**
   - Look for clues: "Form 17C" -> Official. "Is my vote safe?" -> Citizen. "Give me the data" -> Journalist.
   
2. **Determine Interaction Style:**
   - Did they ask for "quick answer"? -> "Fast".
   - Did they ask "explain in detail"? -> "Detailed".
   - Did they ask "don't give me links"? -> Set content preferences accordingly.

3. **Update Content Preferences:**
   - If user says "Stop showing me Twitter links", set `include_twitter_links: false`.
   - If user says "Give me official sources", set `include_source_urls: true`.

4. **Summarize:** But dont make it like chat history, it should be psychological summary of user.
   - Keep a running narrative summary of *what* they are investigating.

5. **Update Name:**
   - If user says "My name is ...", update the name.

6. **Update Location:**
   - If user says "I am from ...", update the location.

### OUTPUT FORMAT (JSON ONLY)
{{  "name": "...",
    "location": "...",
    "persona": "...",
    "interaction_style": "...",
    "content_preferences": {{  //default is true for all.
        "show_twitter": true,
        "show_urls": true,
        "show_actions": true
    }},
    "summary": "Updated narrative summary..."
}}
"""
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", MEMORY_PROMPT),
    ("human", """
    ANALYZE THIS INTERACTION:
    
    --- EXISTING PROFILE ---
    {current_profile}
    
    --- NEW INTERACTION ---
    User: {last_user_msg}
    Agent: {last_agent_msg}
    """)
])

def memory_update_node(state: AgentState, config: RunnableConfig):
    """
    Node 4: The Scribe (Writes Structured LTM)
    """
    # 1. Identity Check
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", "guest")
    if not user_id: return {}
    point_id = get_user_uuid(user_id)
    
    print(f"💾 [MEMORY] Analyzing interaction for '{user_id}'...")

    # 2. Fetch Existing Profile (So we don't overwrite preferences)
    current_profile_str = "New User"
    try:
        points = client.retrieve(
            collection_name="user_profiles",
            ids=[point_id]
        )
        if points:
            # We feed the FULL existing payload so the LLM knows what to keep
            current_profile_str = json.dumps(points[0].payload)
    except Exception:
        pass

    # 3. Prompt the LLM
    last_user = state["messages"][-2].content
    last_agent = state["messages"][-1].content
    
    # Format the prompt
    chain = memory_prompt | llm_memory
    
    # 4. Generate & Parse
    try:
        response = response_msg = chain.invoke({
            "current_profile": current_profile_str,
            "last_user_msg": last_user,
            "last_agent_msg": last_agent
        })
        # Clean Markdown if present
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        new_profile = json.loads(clean_json)
        
        # 5. UPSERT to Qdrant
        # We store the 'summary' as the vector (for semantic search)
        # And the REST as structured payload (for the Agent to read)
        
        summary_text = new_profile.get("summary", "User Profile")
        e5_input = f"passage: {summary_text}"
        
        # 2. Encode
        vector_list = dense_text_model.encode(e5_input).tolist()

        client.upsert(
            collection_name=MEMORY_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={"summary_vector": vector_list},
                    payload={
                        "user_id": user_id,
                        # Overwrite fields with new analysis
                        "name": new_profile.get("name"),
                        "location": new_profile.get("location"),
                        "persona": new_profile.get("persona"),
                        "interaction_style": new_profile.get("interaction_style"),
                        "content_preferences": new_profile.get("content_preferences"),
                        "summary": summary_text,
                        
                    }
                )
            ]
        )
        print(f"   -> ✅ Profile Updated: {new_profile['persona']} | {new_profile['interaction_style']}")
        
    except Exception as e:
        print(f"   -> ❌ Memory Update Failed: {e}")

    return {}