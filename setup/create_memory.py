# setup_memory_db.py
from qdrant_client import models
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

load_dotenv()

print("⏳ [SYSTEM] Initializing AI Models & Database Connection...")

# 1. Database Client (Fast)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "user_profiles"

def setup_user_db():
    print(f"⚙️ Setting up Memory Collection: '{COLLECTION_NAME}'...")
    
    # 1. Check if exists
    if client.collection_exists(COLLECTION_NAME):
        print(f"⚠️ Collection '{COLLECTION_NAME}' already exists.")
        print("   (If you get vector errors, delete this collection and run this script again.)")
        return

    # 2. Create with NAMED Vector (Best Practice)
    # E5-Base produces 768-dimension vectors
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "summary_vector": models.VectorParams(
                size=768, 
                distance=models.Distance.COSINE
            )
        }
    )
    print("✅ Memory Collection Created Successfully!")

if __name__ == "__main__":
    setup_user_db()