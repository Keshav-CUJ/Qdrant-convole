# %%
import os
import json
from dotenv import load_dotenv
import textwrap
from qdrant_client import QdrantClient
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import uuid
import numpy as np
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

# %%
load_dotenv()
MODEL_ID = "intfloat/multilingual-e5-base"
qdrant_api_key = os.getenv("QDRANT_API_KEY")
cluster_endpoint = os.getenv("QDRANT_CLUSTER_ENDPOINT")

# %%
print(f"Loading Tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# %%
MAX_LIMIT = 512

def check_token_counts():

    # 1. Load the JSON Data
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{JSON_FILE}' not found. Please create it first.")
        return

    print(f"Analyzing {len(data)} payloads...\n")
    print("-" * 80)
    print(f"{'IDX':<5} | {'STATUS':<10} | {'COUNT':<8} | {'EXCESS':<8} | {'CONTENT SNIPPET'}")
    print("-" * 80)

    total_overflows = 0

    # 2. Loop through each entry
    for i, item in enumerate(data):
        try:
            # Safely access the nested field
            payload = item.get("payload", {})
            text = payload.get("text_content", "")
            
            if not text:
                print(f"{i:<5} | EMPTY      | 0        | 0        | [No text content]")
                continue

            
            
            text_to_check = f"passage: {text}" 
            tokens = tokenizer.encode(text_to_check, add_special_tokens=True)
            count = len(tokens)
            
            # 4. Calculate Excess
            excess = count - MAX_LIMIT
            
            # Preview (First 40 chars)
            snippet = text[:40].replace("\n", " ") + "..."

            if excess > 0:
                status = "OVER"
                overflow_str = f"+{excess}"
                total_overflows += 1
            else:
                status = "OK"
                overflow_str = "0"

            print(f"{i:<5} | {status:<10} | {count:<8} | {overflow_str:<8} | {snippet}")

        except Exception as e:
            print(f"{i:<5} | ERROR      | -        | -        | {str(e)}")

    print("-" * 80)
    if total_overflows > 0:
        print(f"WARNING: {total_overflows} records exceeded the {MAX_LIMIT} token limit.")
        print("TIP: Use a Semantic Splitter on these specific records.")
    else:
        print("Success! All payloads fit within the context window.")

JSON_FILE = "clean_EVM.json"
check_token_counts()


# %%
JSON_FILE = "clean_FAQ.json"
check_token_counts()

# %%
client = QdrantClient(
    url=cluster_endpoint,
    api_key=qdrant_api_key,
)



# %%
COLLECTION_NAME = "Hybrid_Collection_CONVOLVE"
DENSE_MODEL_NAME = "intfloat/multilingual-e5-base"
SPARSE_MODEL_NAME = "Qdrant/bm25"
IMAGE_MODEL_NAME = "clip-ViT-B-32"

dense_model = SentenceTransformer(DENSE_MODEL_NAME)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
image_model = SentenceTransformer(IMAGE_MODEL_NAME)

DENSE_VECTOR_SIZE = dense_generator.get_sentence_embedding_dimension()
IMAGE_DIM= len(image_generator.encode("test"))

print(f"DENSE_VECTOR_SIZE: {DENSE_VECTOR_SIZE}")
print(f"IMAGE_DIM: {len(image_generator.encode("test"))}")

# %%
# 2. Create Collection with Named Vectors
# We define a specific configuration for EACH vector type


if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense_text": models.VectorParams(
                size=768,  # E5-base is 768 dimensions
                distance=models.Distance.COSINE
            ),
            "dense_image": models.VectorParams(size=IMAGE_DIM, distance=models.Distance.COSINE),
           
        },
        sparse_vectors_config={
            "sparse_text": models.SparseVectorParams(
                modifier=models.Modifier.IDF # Beneficial for BM25
            ),
        }
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

# 2. Create Payload Indexes (The important part)
# You need to specify the field_name and the field_schema (Keyword, Integer, Float, etc.)

# Example: Optimizing for "category" (e.g., "News", "Sports")
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=models.PayloadSchemaType.KEYWORD, # Use KEYWORD for strings
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="topic_tags",
    field_schema=models.PayloadSchemaType.KEYWORD, # Use FLOAT for number ranges
)

# Example: Optimizing for "trust_score" (e.g., filtering score > 0.9)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="trust_score",
    field_schema=models.PayloadSchemaType.FLOAT, # Use FLOAT for number ranges
)

print("Indexes created!")

# %%
def get_semantic_chunks_llama(text_content):
    
    # 1. Initialize the Embedding Model
    # LlamaIndex handles the model loading internally here
    embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

    # 2. Configure the Splitter (Using your exact params)
    text_splitter = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=90, 
        embed_model=embed_model
    )

    # 3. Create a Document Object
    # LlamaIndex splitters expect a list of 'Document' objects, not raw strings
    documents = [Document(text=text_content)]

    # 4. Get Nodes
    # The splitter returns 'Node' objects, which contain metadata and relationships
    nodes = text_splitter.get_nodes_from_documents(documents)

    
    
    return [node.text for node in nodes]



# %%
MAX_TOKEN_LIMIT=512

# %%
def get_token_count(text):
    
    print("Checking token count")

    if not text: return 0
    # E5 expects "passage: " prefix, so we count that too
    text_to_check = f"passage: {text}"
    tokens = tokenizer.encode(text_to_check, add_special_tokens=True)
    return len(tokens)

# %%



def process_record(record):
    original_payload = record.get("payload", {})
    text_content = original_payload.get("debunked_myth", "")
    
    if not text_content: return []

    # --- STEP 1: PRE-CHECK ---
    token_count = get_token_count(text_content)
    chunks = []

    if token_count <= MAX_TOKEN_LIMIT:
        # Case A: Fits in window -> No chunking needed
        print(f"   [OK] {token_count} tokens. Keeping original text.")
        chunks = [text_content]
    else:
        # Case B: Overflow -> Trigger Semantic Splitter
        print(f"   [OVERFLOW] {token_count} tokens. Triggering Semantic Splitter...")
        chunks = get_semantic_chunks_llama(text_content)
        print(f"      -> Split into {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks):
             print(f"--- Chunk {i+1} ---")
             print(chunk)
             print()

   
    dense_inputs = [f"passage: {c}" for c in chunks]
    dense_vectors = dense_model.encode(dense_inputs, normalize_embeddings=True)
    sparse_vectors = list(sparse_model.embed(chunks))

    points = []
    
    # --- STEP 3: CONSTRUCT POINTS ---
    for i, chunk in enumerate(chunks):
        
        # Merge: Original Payload + Chunk Specifics
        # We copy the original payload so we don't modify the source data
        full_payload = original_payload.copy()
        
        # Add chunk-specific overrides
        full_payload.update({
            "text_content": chunk,          # Overwrite text with just this chunk
            # "is_chunked": len(chunks) > 1,  # Flag to know if this was split
            # "chunk_index": i,
            # "Realtiy": text_content  # Optional: Keep original if split
        })

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense_text": dense_vectors[i].tolist(),
                "sparse_text": models.SparseVector(       #Check the Average document length for sparse vectors
                    indices=sparse_vectors[i].indices.tolist(),
                    values=sparse_vectors[i].values.tolist()
                )
            },
            payload=full_payload # <--- Stores ALL fields
        )
        points.append(point)
        print("Created the point for chunk and payload is \n ", full_payload)

    return points



    

# %%
JSON_FILE_PATH = "clean_EVM.json"

with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)


all_points = []
    
print(f"\n🚀 Processing {len(data)} records...")

for idx, record in enumerate(data):
    print(f"Record {idx + 1}:", end="")
    points = process_record(record)
    all_points.extend(points)

    # Batch Upload
if all_points:
    print(f"\n📤 Uploading {len(all_points)} vectors to Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=all_points
    )
    print("✅ Done!")


JSON_FILE_PATH = "clean_FAQ.json"

with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)


all_points = []
    
print(f"\n🚀 Processing {len(data)} records...")

for idx, record in enumerate(data):
    print(f"Record {idx + 1}:", end="")
    points = process_record(record)
    all_points.extend(points)

    # Batch Upload
if all_points:
    print(f"\n📤 Uploading {len(all_points)} vectors to Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=all_points
    )
    print("✅ Done!")




# %%
import json
import uuid
import os
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO

# %%

def get_image_embedding(image_source):
    if not image_source: return None
    
    try:
        img = None
        
        # FIX 1: Check for both http and https
        if image_source.startswith(("http://", "https://")):
            response = requests.get(image_source, stream=True, timeout=10)
            
            # FIX 2: Raise error if status is 404/500
            response.raise_for_status() 
            
            img = Image.open(BytesIO(response.content))
        
        elif os.path.exists(image_source):
            img = Image.open(image_source)

        if img:
            # FIX 3: Force conversion to RGB (Fixes PNG/RGBA errors)
            img = img.convert("RGB")
            
            # Generate vector
            return image_model.encode(img, normalize_embeddings=True).tolist()
            
    except Exception as e:
        print(f"⚠️ Image Error processing '{image_source}': {e}")
        
    return None


# %%
# --- 2. HELPER FUNCTIONS ---

def construct_text_from_visual_payload(payload):
    """
    Combines Title + Concepts + Description into one searchable text block.
    """
    parts = []
    if payload.get("title"): parts.append(payload["title"])
    if payload.get("visual_concepts"): parts.append(", ".join(payload["visual_concepts"]))
    if payload.get("description"): parts.append(payload["description"])
    return ". ".join(parts)

# %%
# --- 3. PROCESS VISUAL RECORDS ---

def process_visual_record(record):
    payload = record.get("payload", {})
    
    # Strictly process only visual records
    if payload.get("record_type") != "official_visual_truth":
        return []

    print(f"Processing: {payload.get('title', 'Untitled')}")

    # A. Generate Image Vector (Lane 1)
    image_url = payload.get("image_url")
    image_vector = get_image_embedding(image_url)
    
    if image_vector is None:
        print("   -> Skipping (No Image)")
        return []

    # B. Generate Text Content (Lane 2 & 3)
    # We construct a rich text block so you can search for this image via text
    rich_text = construct_text_from_visual_payload(payload)
    
    # C. Chunking (Because descriptions can be long)
    # We reuse the logic: If short, 1 chunk. If long, split it.
    
     
    # --- STEP 1: PRE-CHECK ---
    token_count = get_token_count(rich_text)
    chunks = []

    if token_count <= MAX_TOKEN_LIMIT:
        # Case A: Fits in window -> No chunking needed
        print(f"   [OK] {token_count} tokens. Keeping original text.")
        chunks = [rich_text]
    else:
        # Case B: Overflow -> Trigger Semantic Splitter
        print(f"   [OVERFLOW] {token_count} tokens. Triggering Semantic Splitter...")
        chunks = get_semantic_chunks_llama(rich_text)
        print(f"      -> Split into {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks):
             print(f"--- Chunk {i+1} ---")
             print(chunk)
             print()


    # D. Create Text Vectors
    dense_inputs = [f"passage: {c}" for c in chunks]
    dense_vectors = dense_model.encode(dense_inputs, normalize_embeddings=True)
    sparse_vectors = list(sparse_model.embed(chunks))

    points = []
    for i, chunk in enumerate(chunks):
        
        full_payload = payload.copy()
        full_payload.update({
            "text_content": chunk, # Stores the specific text chunk
            "original_full_text": rich_text if len(chunks) > 1 else None,
            "chunk_index": i
        })

        # E. Create Point with ALL THREE Vectors
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense_image": image_vector,      # <--- CLIP Vector
                "dense_text": dense_vectors[i].tolist(), # <--- E5 Vector
                "sparse_text": models.SparseVector(      # <--- BM25 Vector
                    indices=sparse_vectors[i].indices.tolist(),
                    values=sparse_vectors[i].values.tolist()
                )
            },
            payload=full_payload
        )
        points.append(point)
        
    return points



    

 
JSON_FILE_PATH="metadata.json"

with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_points = []
    
for record in data:
    new_points = process_visual_record(record)
    all_points.extend(new_points)

if all_points:
    print(f"\n📤 Uploading {len(all_points)} fully hybrid visual points...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=all_points
    )
    print("✅ Upload Complete.")
else:
    print("No visual records processed.")


