from fastapi import FastAPI, UploadFile, File
from src.retrieval import hybrid_retrieve
from src.generation import generate_reasoned_answer
from src.ingestion import ingest_pdf
from pydantic import BaseModel
import shutil
import os

app = FastAPI(title="Gold Standard RAG")

class QueryRequest(BaseModel):
    query: str

# @app.get("/")
# async def root():
#     return {
#         "message": "Gold Standard RAG API",
#         "docs": "http://localhost:8000/docs",
#         "endpoints": {
#             "POST /ask": "Submit a query for retrieval and generation"
#         }
#     }

@app.post("/upload-pdf")
async def upload_document(file: UploadFile = File(...)):
    # Create a temporary path to save the uploaded file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run our ingestion pipeline
        ingest_pdf(temp_path)
        return {"message": f"Successfully processed {file.filename}"}
    finally:
        # Clean up the file after indexing
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Process a query and return reasoned answer with sources."""
    try:
        # 1. Retrieve
        context_chunks = hybrid_retrieve(request.query)
        
        # 2. Generate
        result = generate_reasoned_answer(request.query, context_chunks)
        
        return {
            "query": request.query,
            "reasoning": result["thought"],
            "answer": result["answer"],
            "sources_used": len(context_chunks)
        }
    except Exception as e:
        print(f"Error processing query: {e}")
        return {
            "query": request.query,
            "reasoning": "An error occurred",
            "answer": f"Error processing query: {str(e)}",
            "sources_used": 0,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)