from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from config import settings
from src.models import get_embed_model, get_rerank_model

q_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY or None,
)

def ensure_collection_exists():
    """Ensure the collection exists in Qdrant with required indexes."""
    try:
        q_client.get_collection(settings.COLLECTION_NAME)
        print(f"Collection {settings.COLLECTION_NAME} already exists")
    except Exception as e:
        print(f"Collection doesn't exist, creating it...")
        try:
            q_client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=get_embed_model().get_sentence_embedding_dimension(),
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

    # Ensure payload index exists for session filtering (required by Qdrant Cloud)
    try:
        q_client.create_payload_index(
            collection_name=settings.COLLECTION_NAME,
            field_name="session_id",
            field_schema="keyword",
        )
        print("Created payload index on session_id")
    except Exception as idx_err:
        if "already exists" in str(idx_err).lower() or "409" in str(idx_err):
            pass  # index already exists
        else:
            print(f"Index creation note: {idx_err}")

def hybrid_retrieve(query: str, top_k: int = 5, session_id: str = ""):
    """Retrieve context chunks using hybrid search, filtered by session."""
    try:
        ensure_collection_exists()
        
        query_vec = get_embed_model().encode(query).tolist()
        
        # Filter by session_id so users only see their own data
        query_filter = None
        if session_id:
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )]
            )
        
        hits = q_client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_vec,
            query_filter=query_filter,
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
        scores = get_rerank_model().predict([(query, p) for p in passages])
        
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