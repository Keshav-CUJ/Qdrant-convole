import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (mas_election_agent/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to Python's search path
sys.path.append(parent_dir)

from src.tools.qdrant_search import search_hybrid, search_sparse, search_image

# --- CONFIGURATION ---
# Replace with a real image path you have locally for testing
TEST_IMAGE_PATH = "https://github.com/Keshav-CUJ/Qdrant-convole/raw/main/images/EVMbackpack.png" 

def test_text_search():
    print("\n🧪 TEST 1: Hybrid Search (Concepts)")
    print("-" * 40)
    query = "What is the cost of VVPAT machine?"
    
    # We expect to find the price document
    results = search_hybrid(query, limit=2)
    
    for pt in results:
        print(f"✅ Found: {pt.payload.get('source_url')} | Score: {pt.score:.3f}")
        print(f"   Snippet: {pt.payload.get('text_content')[:100]}...")

def test_sparse_search():
    print("\n🧪 TEST 2: Sparse Search (Specific Keywords)")
    print("-" * 40)
    query = "Form 17C" 
    
    # We expect to find the specific form protocol
    results = search_sparse(query, limit=2)
    
    for pt in results:
        print(f"✅ Found: {pt.payload.get('text_content')} | Score: {pt.score:.3f}")

def test_image_search():
    print("\n🧪 TEST 3: Image Search (Visual)")
    print("-" * 40)
    
    

    # We expect to find the VVPAT or EVM object
    results = search_image(TEST_IMAGE_PATH, filters={"category": "EVM and VVPAT"}, limit=2)
    
    for pt in results:
        print(f"✅ Found: {pt.payload.get('title')} | Score: {pt.score:.3f}")

def test_filtered_search():
    print("\n🧪 TEST 4: Filtered Search (Metadata)")
    print("-" * 40)
    query = "EVM can be hacked?"
    
    # We only want "Official Truth", no myths
    filters = {
        "category": "Busted fake news",
        "topic_tags": ["Fake news"] # Should match if this tag exists
    }
    
    results = search_hybrid(query, filters=filters, limit=2)
    
    if not results:
        print("❌ No results found with these filters.")
    else:
        for pt in results:
            print(f"✅ Found: {pt.payload.get('source_url')} | Category: {pt.payload.get('category')}")

if __name__ == "__main__":
    test_text_search()
    test_sparse_search()
    test_filtered_search()
    test_image_search()