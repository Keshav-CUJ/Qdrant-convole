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

# 2. AI Models (Slow - Loads only once on import)
# We use a global variable pattern to ensure they persist
dense_text_model = SentenceTransformer("intfloat/multilingual-e5-base")
dense_image_model = SentenceTransformer("clip-ViT-B-32")
sparse_text_model = SparseTextEmbedding(model_name="Qdrant/bm25")

DATA_COLLECTION_NAME = "Hybrid_Collection_CONVOLVE"
MEMORY_COLLECTION_NAME = "user_profiles"


print("✅ [SYSTEM] Models Loaded Successfully.")