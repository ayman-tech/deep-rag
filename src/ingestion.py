from openai import OpenAI
from qdrant_client import QdrantClient, models
from config import settings
from pypdf import PdfReader
from src.models import get_embed_model
import os

# OCR imports — optional locally, required in Docker
try:
    import pypdfium2 as pdfium
    import pytesseract
    # Auto-detect Tesseract on Windows if not in PATH
    import platform
    if platform.system() == "Windows":
        _win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(_win_path):
            pytesseract.pytesseract.tesseract_cmd = _win_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  OCR dependencies not available. Image-based PDFs won't be processed.")

client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
q_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY or None,
)

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

def add_to_index(texts: list, metadata: dict = None, session_id: str = ""):
    """Adds text chunks to the Qdrant vector database."""
    if metadata is None:
        metadata = {}
    
    for i, text in enumerate(texts):
        try:
            vector = get_embed_model().encode(text).tolist()
            point_id = hash(text + session_id) % (10 ** 8)
            
            payload = {
                "original_text": text,
                "context": metadata.get("context", ""),
                "session_id": session_id,
                **metadata
            }
            
            # Use upsert which works across versions
            q_client.upsert(
                collection_name=settings.COLLECTION_NAME,
                wait=True,
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

def _ocr_page(pdf_bytes: bytes, page_index: int) -> str:
    """Render a single PDF page to image and OCR it."""
    if not OCR_AVAILABLE:
        print(f"OCR skipped for page {page_index + 1}: dependencies not installed")
        return ""
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        page = pdf[page_index]
        bitmap = page.render(scale=2)  # 2x for better OCR accuracy
        pil_image = bitmap.to_pil()
        text = pytesseract.image_to_string(pil_image)
        pdf.close()
        return text.strip()
    except Exception as e:
        print(f"OCR failed on page {page_index + 1}: {e}")
        return ""


def ingest_pdf(file_path: str, session_id: str = ""):
    """
    Extracts text from a PDF and processes it using 
    Contextual Chunking + Vector Indexing.
    """
    try:
        # Read entire file into memory so the file handle is released immediately
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        from io import BytesIO
        reader = PdfReader(BytesIO(pdf_bytes))
        full_text = ""
        
        # 1. Extract all text to create a 'Document Summary' for context
        for page in reader.pages:
            full_text += page.extract_text()
        
        # 2. Iterate through pages to create chunks
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text().strip()
            
            # Fallback to OCR if less text extracted (image-based PDF)
            if len(page_text) < 150:
                print(f"Page {page_num + 1}: less/no text found, running OCR...")
                page_text = _ocr_page(pdf_bytes, page_num)
            
            if not page_text:
                print(f"Page {page_num + 1}: skipped (empty even after OCR)")
                continue
                
            # 3. Apply Contextual Chunking (The DeepSeek part)
            context = get_contextual_header(full_text, page_text)
            final_smart_chunk = f"{context}\n\n{page_text}" if context else page_text
            
            # 4. Save to Vector DB with page metadata
            add_to_index([final_smart_chunk], metadata={"page": page_num + 1}, session_id=session_id)
        
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