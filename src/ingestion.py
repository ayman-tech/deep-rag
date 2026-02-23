import torch
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from config import settings
from pypdf import PdfReader

client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
q_client = QdrantClient(settings.QDRANT_URL)
embed_model = SentenceTransformer(settings.DENSE_MODEL)

def get_contextual_header(full_doc: str, chunk: str) -> str:
    """Situates a chunk within its document context (Anthropic method)."""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": f"Situate this chunk in the doc: <doc>{full_doc[:2000]}</doc>\n<chunk>{chunk}</chunk>"}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting contextual header: {e}")
        return ""

def add_to_index(texts: list, metadata: dict = None):
    """Adds text chunks to the Qdrant vector database."""
    if metadata is None:
        metadata = {}
    
    for i, text in enumerate(texts):
        try:
            vector = embed_model.encode(text).tolist()
            point_id = hash(text) % (10 ** 8)
            
            payload = {
                "original_text": text,
                "context": metadata.get("context", ""),
                **metadata
            }
            
            # Use upsert which works across versions
            q_client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=[models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )]
            )
            print(f"Added point {point_id} to index")
        except Exception as e:
            print(f"Error adding chunk {i} to index: {e}")
            import traceback
            traceback.print_exc()
            continue

def ingest_pdf(file_path: str):
    """
    Extracts text from a PDF and processes it using 
    Contextual Chunking + Vector Indexing.
    """
    try:
        reader = PdfReader(file_path)
        full_text = ""
        
        # 1. Extract all text to create a 'Document Summary' for context
        for page in reader.pages:
            full_text += page.extract_text()
        
        # 2. Iterate through pages to create chunks
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text().strip()
            
            if not page_text:
                continue
                
            # 3. Apply Contextual Chunking (The DeepSeek part)
            context = get_contextual_header(full_text, page_text)
            final_smart_chunk = f"{context}\n\n{page_text}" if context else page_text
            
            # 4. Save to Vector DB with page metadata
            add_to_index([final_smart_chunk], metadata={"page": page_num + 1})
        
        print(f"Successfully indexed: {file_path}")
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise

def process_and_index(full_text: str):
    """Process text into chunks and index them."""
    # 1. Simple split (production would use RecursiveCharacterTextSplitter)
    chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 800)]
    
    for chunk in chunks:
        context = get_contextual_header(full_text, chunk)
        final_text = f"{context}\n\n{chunk}" if context else chunk
        add_to_index([final_text], metadata={"context": context})