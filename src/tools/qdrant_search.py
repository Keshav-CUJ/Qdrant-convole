
import os
import json
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from PIL import Image
import requests
from io import BytesIO

from src.config import (
    client, 
    dense_text_model, 
    dense_image_model, 
    sparse_text_model,
    DATA_COLLECTION_NAME
)




# --- CONFIGURATION ---
COLLECTION_NAME = DATA_COLLECTION_NAME

# --- 2. HELPER: DYNAMIC FILTER BUILDER ---
def build_filter(filter_dict):
    """
    Converts a simple dictionary {'category': 'News', 'trust_score': 0.9}
    into a complex Qdrant Filter object.
    """
    if not filter_dict: return None
    
    conditions = []
    for key, value in filter_dict.items():
        # Handle simple equality (Category = "News")
        if isinstance(value, str):
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
        # Handle ranges (Score >= 0.9)
        elif isinstance(value, (int, float)):
             conditions.append(
                models.FieldCondition(key=key, range=models.Range(gte=value))
            )
        # Handle lists (Category IN ["News", "Reports"])
        elif isinstance(value, list):
             conditions.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=value))
            )

    return models.Filter(must=conditions)


def search_sparse(query_text, filters=None, limit=5):
    print(f"\n🔍 [SPARSE] Searching for: '{query_text}'")
    
    query_vector = list(sparse_text_model.embed([query_text]))[0]

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.SparseVector(
            indices=query_vector.indices.tolist(),
            values=query_vector.values.tolist()
        ),
        using="sparse_text",    # Specify the vector name here
        query_filter=build_filter(filters),
        limit=limit
    ).points
    
    return hits


# --- 5. RETRIEVAL FUNCTION 3: IMAGE SEARCH (Visual) ---
def search_image(image_source, filters=None, limit=5):
    print(f"\n🔍 [IMAGE] Searching with image input...")
    
    if not image_source: return []
    
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
            
   

        # 2. Vectorize Image (CLIP)
        image_vector = dense_image_model.encode(img, normalize_embeddings=True).tolist()

        # 3. Search "dense_image" vector space
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=image_vector,
            using="dense_image",    # Specify the vector name here
            query_filter=build_filter(filters),
            limit=limit
        ).points

        return hits

    except Exception as e:
        print(f"❌ Image Search Failed: {e}")
        return []


# --- 6. RETRIEVAL FUNCTION 4: HYBRID SEARCH (RRF Fusion) ---
def search_hybrid(query_text, filters=None, limit=5):
    print(f"\n🔍 [HYBRID] Searching for: '{query_text}'")
    
    # RRF (Reciprocal Rank Fusion) is the industry standard for 
    # combining Dense (Semantic) + Sparse (Keyword) results.
    
    # 1. Get Results from both worlds
    dense_hits = search_dense(query_text, filters, limit=limit*2)
    sparse_hits = search_sparse(query_text, filters, limit=limit*2)
    
    # 2. Fuse Scores (RRF Algorithm)
    # Score = 1 / (rank + k)
    rank_k = 60
    fused_scores = {}
    
    # Process Dense Ranks
    for rank, hit in enumerate(dense_hits):
        if hit.id not in fused_scores: fused_scores[hit.id] = {"hit": hit, "score": 0}
        fused_scores[hit.id]["score"] += 1 / (rank + rank_k)
        
    # Process Sparse Ranks
    for rank, hit in enumerate(sparse_hits):
        if hit.id not in fused_scores: fused_scores[hit.id] = {"hit": hit, "score": 0}
        fused_scores[hit.id]["score"] += 1 / (rank + rank_k)
    
    # 3. Sort by new fused score
    sorted_results = sorted(
        fused_scores.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )
    
    # Return top N original hit objects
    return [item["hit"] for item in sorted_results[:limit]]

# --- 3. RETRIEVAL FUNCTION 1: DENSE SEARCH (Semantic) ---
def search_dense(query_text, filters=None, limit=5):
    print(f"\n🔍 [DENSE] Searching for: '{query_text}'")
    
    # 1. Vectorize Query (E5 needs "query: " prefix)
    query_vector = dense_text_model.encode(
        f"query: {query_text}", 
        normalize_embeddings=True
    ).tolist()

    # 2. Search "dense_text" vector space
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,     # Pass the vector list directly
        using="dense_text",     # Specify the vector name here
        query_filter=build_filter(filters),
        limit=limit
    ).points
    return hits


# # A. Test Dense (Semantic Question)
# results = search_dense(
#     "The EVM and VVPAT are free in price", 
#     filters={"topic_tags": [
#         "EVM Price",
#         "VVPAT Price"
#       ]} 
# )
# for hit in results:
#     print(f"   Score: {hit.score:.3f} | {hit.payload.get('text_content')[:]}...")





# # B. Test Sparse (Specific Keywords)
# results = search_sparse(
#     "EVM can be hacked", 
#     filters={"topic_tags": [
#         "Election scams",
#         "Lies around elections"
#       ], "trust_score": 0.9}  
# )
# for hit in results:
#     print(f"   Score: {hit.score:.3f} | {hit.payload.get('text_content', 'No Title')}")

# # C. Test Image (Visual Match)
# # Use a dummy image or a real path if you have one
# # results = search_image("assets/test_backpack.jpg") 

# # D. Test Hybrid (Best of both worlds)



# results = search_hybrid(
#     "Mandatory verification of VVPAT slips of randomly selected 05 polling stations per Assembly Constituency",
#     filters={"trust_score": 1.0}
# )
# print("\n🏆 HYBRID WINNERS:")
# for hit in results:
#     # Note: Hybrid returns the original Qdrant object, but score is RRF score now
#     print(f"   Payload: {hit.payload.get('title') or hit.payload.get('text_content')[:40]}")


# # Use a direct image link (ending in .jpg or .png)
# query_image_url = "https://github.com/Keshav-CUJ/Qdrant-convole/raw/main/images/EVMbackpack.png"

# results = search_image(
#     image_source=query_image_url, 
#     limit=3
# )

# print(f"\n🌐 Visual Search Results for URL:")
# for hit in results:
#     print(f"   Score: {hit.score:.3f} | Found: {hit.payload.get('title')}")


