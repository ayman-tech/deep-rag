from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import CrossEncoder, SentenceTransformer
from config import settings

q_client = QdrantClient(settings.QDRANT_URL)
embed_model = SentenceTransformer(settings.DENSE_MODEL)
rerank_model = CrossEncoder(settings.RERANK_MODEL)

def ensure_collection_exists():
    """Ensure the collection exists in Qdrant."""
    try:
        q_client.get_collection(settings.COLLECTION_NAME)
        print(f"Collection {settings.COLLECTION_NAME} already exists")
    except Exception as e:
        print(f"Collection doesn't exist, creating it...")
        try:
            q_client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=embed_model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE
                ),
            )
            print(f"Collection {settings.COLLECTION_NAME} created successfully")
        except Exception as create_error:
            # 409 Conflict means collection already exists - this is OK
            if "409" in str(create_error) or "already exists" in str(create_error):
                print(f"Collection {settings.COLLECTION_NAME} already exists (no action needed)")
            else:
                print(f"Error creating collection: {create_error}")

def hybrid_retrieve(query: str, top_k: int = 5):
    """Retrieve context chunks using hybrid search."""
    try:
        ensure_collection_exists()
        
        query_vec = embed_model.encode(query).tolist()
        
        # Use search_points for better compatibility
        hits = q_client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_vec,
            limit=20,
        )
        
        # 2. Reranking (ColBERT-style Precision)
        passages = []
        for hit in hits:
            if hasattr(hit, 'payload'):
                context = hit.payload.get("context", "")
                original_text = hit.payload.get("original_text", "")
                passages.append(context + original_text)
        
        if not passages:
            print("No documents found in vector store")
            return ["No relevant documents found. Please upload a PDF first using /upload-pdf endpoint."]
        
        print(f"Retrieved {len(passages)} passages, reranking...")
        scores = rerank_model.predict([(query, p) for p in passages])
        
        # Combine and sort
        results = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        final_results = [r[0] for r in results[:top_k]]
        print(f"Returning top {len(final_results)} results")
        return final_results
    
    except AttributeError as ae:
        print(f"Qdrant Client AttributeError: {ae}")
        print("Checking available methods...")
        print(f"Available methods: {[m for m in dir(q_client) if not m.startswith('_')]}")
        return ["Error accessing vector database. Please check Qdrant connection."]
    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
        return ["Error retrieving documents. Please try again."]