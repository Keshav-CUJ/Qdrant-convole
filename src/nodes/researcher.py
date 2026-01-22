from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.state import AgentState
import json
from src.tools.qdrant_search import search_hybrid, search_image, search_sparse, search_dense
from dotenv import load_dotenv
import os
load_dotenv()
# Initialize LLM (Ensure you have OPENAI_API_KEY in .env)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
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


QUERY_GEN_SYSTEM_PROMPT = """You are an expert information retrieval agent, for Misinformation Detection System.

Your goal is to analyze the user's request and the user's context to construct the PERFECT database search plan, in which you will be calling different tools to get information, which will be used to answer the user's query.

1. Which tool to use.
2. What will be the input for the tool (It can be 'modified user query' or 'image path').
3. Filters to get the more accurte information (e.g. 
4. What is the purpose of the tool.

### 1. AVAILABLE TOOLS
- **search_hybrid**: (Default) Best for general questions and claims. Combines semantic meaning + keywords.
- **search_dense**:  Best for more accurate answers. Uses brute force search.
- **search_sparse**: Best for finding specific codes, form numbers (e.g., "Form 17C"), or exact acronyms.
- **search_image**: MUST be used if the user provides an image path or URL. Finds visually similar records.

### 2. DATABASE SCHEMA (Payload Fields for Filtering)

You can filter by these fields. DO NOT invent new fields.

- **category** (str): "Busted fake news", "Information on EVM & VVPAT", "Information on eligiblity to vote", "Vote Counting essential", "Polling Station essential". (You have only these categories)
- **topic_tags** (list): ["Election scams", "Lies around elections", "eligiblity for challenged voters", "Steps in conduct of Election", "EVM manufacturing", "EVM security", "VVPAT hacking"]. (You can make other tags also).

### 3. FILTERING RULES
- **List Logic:** If filtering by `topic_tags`, the value should be the specific tag string. Qdrant handles the list matching automatically.
- **Complex Filters:** You can apply multiple filters at once but remember they will be ANDed together (e.g., Category AND topic_tags).
- **Context Awareness:** If `user_context` says "User is Presiding Officer", prioritize `category="Polling Process"`.



### OUTPUT FORMAT (JSON LIST)
Output a LIST of tool calls.

In which each entry would be like:
{{
    "tool": "search_hybrid" | "search_sparse" | "search_image" | "search_dense",
    "query": "The optimized search string",
    "filters": {{ "field_name": "value", "field_name_2": ["value1", "value2"] }} or null,
    "purpose": "What is the purpose of this search?"
}}


### STRATEGY FOR "IMAGE + CLAIM"
If the user uploads an image AND makes a claim (e.g., "This machine is hacked"), you MUST generate TWO search plans:
1. **Visual Truth:** A `search_image` to identify what the object *actually* is.
2. **Claim Check:** A `search_hybrid` to find facts about the user's text claim (myths, news, rules).

[
    {{
        "tool": "search_image",
        "query": "path/to/image.jpg", //It will be used as path so please keep the input path of image.
        "filters": {{"category": "Vote Counting essential"}}, //in case of image you have only one filter i.e. category
        "purpose": "Identify the object"
    }},
    {{
        "tool": "search_hybrid",
        "query": "EVM hacking bluetooth 2024",
        "filters": {{"category": "Busted fake news", "topic_tags": ["Election scams"]}}, //in case of text you can use multiple filters
        "purpose": "Verify the text claim"
    }}
]

### 5. OTHER EXAMPLES

**User:** "How much does a VVPAT cost?"
**Output:**
[{{
    "tool": "search_hybrid",
    "query": "VVPAT unit price cost",
    "filters": {{ "topic_tags": ["EVM Price"] }},
    "purpose": "Find the price of VVPAT"
}}]

**User:** "Find this specific seal." (User uploads image: 'assets/seal.png')
**Output:**
[{{
    "tool": "search_image",
    "query": "assets/seal.png", // as inputed
    "filters": {{}},
    "purpose": "Find the seal"
}}]

**User:** "Show me official protocols for Form 17C."
**Output:**
[{{
    "tool": "search_sparse",
    "query": "Form 17C protocol",
    "filters": {{ "category": "Polling Process" }},
    "purpose": "Find the protocol for Form 17C"
}}]

AND ONE LAST THING IF USER DO NOT UPLOAD ANY IMAGE THEN PLEASE DO NOT CALL search_image tool at any random path. AVOID USING FILTERS IN ALMOST EVERY CASE.
"""


# We need to explicitly tell the LLM if an image exists in the variable input
QUERY_GEN_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", QUERY_GEN_SYSTEM_PROMPT),
    ("user", "USER CONTEXT: {user_context}\n\nUSER INPUT: {last_message}\n\nIMAGE UPLOADED: {image_path}")
])

# --- THE NODE FUNCTION ---
def query_gen_node(state: AgentState):
    """
    Node 1: Query Generator
    Decides the plan (Tools + Queries + Filters)
    """
    # 1. Extract inputs
    last_message = state["messages"][-1].content
    user_context = state.get("user_context", "General User")
    # Check if an image path exists in state (passed from UI)
    image_path = state.get("current_image_path", "None") 
    
    print(f"\n🧠 [GEN_QUERY] Analyzing: '{last_message[:30]}...' | Image: {image_path}")
    
    # 2. Invoke Chain
    # We pass the image_path so the LLM knows to trigger 'search_image'
    chain = QUERY_GEN_PROMPT_TEMPLATE | llm | StrOutputParser()
    
    raw_response = chain.invoke({
        "last_message": last_message,
        "user_context": user_context,
        "image_path": image_path
    })
    
    # 3. IMPROVEMENT: Robust JSON Parsing
    try:
        # Strip markdown code blocks if the LLM adds them
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        plans = json.loads(clean_json)
        
        # Safety check: Ensure it's a list
        if isinstance(plans, dict): plans = [plans]
            
        print(f"   -> Generated {len(plans)} Search Plans.")
        
    except json.JSONDecodeError:
        print("   ❌ JSON Error. Fallback to default hybrid search.")
        plans = [{
            "tool": "search_hybrid", 
            "query": last_message, 
            "filters": None,
            "purpose": "Fallback search"
        }]
    
    # 4. Return the correct key 'search_plans'
    return {"search_plans": plans}


def search_execution_node(state: AgentState):
    plans = state.get("search_plans", [])
    combined_results = []
    
    print(f"🕵️ Executing {len(plans)} parallel searches...")

    for i, plan in enumerate(plans):
        tool = plan.get("tool")
        query = plan.get("query")
        filters = plan.get("filters") # Capture the filters!
        purpose = plan.get("purpose", "General Search")
        
        print(f"   [{i+1}] Tool: {tool} | Filters: {filters}")
        
        results = ""
        
        # --- IMPROVEMENT: Handle ALL tools defined in prompt ---
        try:
            if tool == "search_image":
                # Ensure your tool accepts 'filters' argument
                results = search_image(image_source=query, filters=filters)
                
            elif tool == "search_sparse":
                results = search_sparse(query_text=query, filters=filters)
                
            elif tool == "search_dense":
                results = search_hybrid(query_text=query, filters=filters)
                
            else: # Default 'search_hybrid'
                results = search_hybrid(query_text=query, filters=filters)
                
        except Exception as e:
            results = f"Error executing {tool}: {str(e)}"

        # Label the results clearly
        combined_results.append(
            f"=== STEP {i+1}: {purpose} ===\n"
            f"TOOL: {tool}\n"
            f"RESULTS:\n{results}\n"
            f"=========================================\n"
        )
    
    # Join everything
    final_docs = "\n".join(combined_results)
    return {"retrieved_docs": final_docs}

    ######################################################################################################

    # src/nodes/researcher.py (Append this)

# --- RESPONDER PROMPT ---
# src/nodes/researcher.py

# ... (Previous imports & Query Gen Node) ...

# --- THE RESPONDER PROMPT ---
# src/nodes/researcher.py

RESPONDER_SYSTEM_PROMPT = """You are the Final Responder and Recommander agent for a Misinformation Detection System.

YOUR ROLE:
Analyze the User's Query and the Retrieved Evidence to provide a **Verdict**, **Explanation**, and **Actionable Recommendation**, remember to use the user_context to provide a more personalized response.

### INPUT DATA
- **User Query:** {user_query}
- **User Context:** {user_context}
- **Evidence:** {retrieved_docs}



### RESPONSE GUIDELINES

2. **The Verdict (Start with this):**
   - classify the claim as: 🟢 **VERIFIED**, 🔴 **MISINFORMATION**, 🟡 **MISLEADING/OUT OF CONTEXT**, or ⚪ **UNVERIFIED**.
   - If the user uploaded an image, explicitly state: "Visual analysis matches/does not match..."

3. **The Evidence (The "Why"):**
   - Summarize the retrieved facts but keep the main information, **dont compromise for quality**.
   - **MANDATORY:** For deciding to showing URLs, SOURCE you will find content_preference in the user_context like this :
     
     {{
   "show_twitter":true
    "show_urls":true
   "show_actions":true}}
    BUT IF THIS IS MISSING WHICH MEANS USER IS NEW SO SHOW ALL EVIDENCES ANF SORCES URL AND NAMES AND TITLES UNTILL USER MENTIONED EXPLICITLY.                       
   
   - **MANDATORY:**    YOU WILL DEFINETLY FIND EVIDECE AND SOURCE URL AND SOURCE NAME IN THE EVIDENCE, PUT THERE LINK IN RESPONSE UNTILL UNLESS USER SET VALUES AS FALSE FOR THEM IN content_preference.
   - **MANDATORY:** In the evidence_media you can have video, image or tweet so share it with user with proper info about it UNTILL UNLESS USER SET VALUES AS FALSE FOR THEM IN content_preference.
  
   - Example: *"According to the ECI Manual (Source: eci.gov.in)..."*

4. **Actionable Recommendation:**
   - Look for the `actionable_intent` field in the evidence (e.g., "Spread_correct_info", "Report_to_authorities").
   - Tell the user what to do: *"Since this is official truth, please share this information."* or *"This is a known myth. Do not forward it."*

5. **Tone:**
   - Authoritative yet helpful.
   - If the user is an official (based on context), provide technical references (Form numbers, Rules).

6. **Further Action/ Evaluation:**
   - You can ask user to provide more information if needed.
   - You can ask user about his small details for personalization because we have "persona" field in user_context.
   
    IF YOU DONT GET ANY PROPER EVIDENCE THEN RETURN "UNVERIFIED" DONT HALLUCINATE.
"""

responder_prompt = ChatPromptTemplate.from_messages([
    ("system", RESPONDER_SYSTEM_PROMPT),
    ("user", "Here is the data. Give me the verdict.")
])
def responder_node(state: AgentState):
    """
    Node 3: The Writer
    Synthesizes User Query + Evidence -> Final Answer
    """
    # 1. Fetch ALL necessary context from state
    user_query = state["messages"][-1].content  # The User's original text
    docs = state.get("retrieved_docs", "No evidence found.")
    context = state.get("user_context", "General User")
    
    print(f"✍️ [RESPONDER] Synthesizing answer for query: '{user_query[:20]}...'")
    
    # 2. Run the LLM Chain
    chain = responder_prompt | llm
    
    response_msg = chain.invoke({
        "user_query": user_query,    # Pass query explicitly
        "retrieved_docs": docs,      # Pass evidence
        "user_context": context      # Pass profile
    })
    
    # 3. Return the Final Answer
    # LangGraph automatically appends this to the message history
    return {"messages": [response_msg]}