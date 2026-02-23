# 🧠 DeepRAG — Hybrid Retrieval-Augmented Generation System

An end-to-end RAG pipeline with hybrid dense/sparse retrieval, cross-encoder reranking, contextual chunking, and DeepSeek LLM generation — served via a full-stack NiceGUI web interface.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NiceGUI Frontend (:8000)                     │
│                   Upload PDF  |  Ask Questions  |  View Results     │
└──────────────┬──────────────────────────────────┬───────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────┐       ┌──────────────────────────────────┐
│    Ingestion Pipeline    │       │        Query Pipeline            │
│                          │       │                                  │
│  ┌────────────────────┐  │       │  ┌────────────────────────────┐  │
│  │  PDF Text Extract  │  │       │  │   Dense Embedding (BGE-M3) │  │
│  │     (pypdf)        │  │       │  └─────────────┬──────────────┘  │
│  └────────┬───────────┘  │       │                │                 │
│           ▼              │       │                ▼                 │
│  ┌────────────────────┐  │       │  ┌────────────────────────────┐  │
│  │ Contextual Chunking│  │       │  │   Hybrid Search (Qdrant)   │  │
│  │   (DeepSeek LLM)   │  │       │  │   Top-20 candidates       │  │
│  └────────┬───────────┘  │       │  └─────────────┬──────────────┘  │
│           ▼              │       │                │                 │
│  ┌────────────────────┐  │       │                ▼                 │
│  │  Dense Embedding   │  │       │  ┌────────────────────────────┐  │
│  │   (BGE-M3)         │  │       │  │  Cross-Encoder Reranking   │  │
│  └────────┬───────────┘  │       │  │  (BGE-Reranker-v2-M3)     │  │
│           ▼              │       │  └─────────────┬──────────────┘  │
│  ┌────────────────────┐  │       │                │                 │
│  │  Qdrant Upsert     │  │       │                ▼                 │
│  │  (Vector Indexing)  │  │       │  ┌────────────────────────────┐  │
│  └────────────────────┘  │       │  │  DeepSeek LLM Generation   │  │
└──────────────────────────┘       │  │  (Chain-of-Thought)        │  │
                                   │  └────────────────────────────┘  │
                                   └──────────────────────────────────┘
                                              │
                                   ┌──────────┴──────────┐
                                   │   Qdrant (Docker)    │
                                   │   localhost:6333     │
                                   └─────────────────────┘
```

## ⚙️ Tech Stack

| Component           | Technology                          |
| ------------------- | ----------------------------------- |
| **Frontend**        | NiceGUI (Quasar/Vue-based)          |
| **Backend**         | FastAPI                             |
| **Vector Database** | Qdrant (Docker)                     |
| **Embeddings**      | BAAI/bge-m3 (Sentence Transformers) |
| **Reranker**        | BAAI/bge-reranker-v2-m3             |
| **LLM**             | DeepSeek Chat API                   |
| **PDF Parsing**     | pypdf                               |

## ✨ Features

- 🔗 **Contextual Chunking** — Each chunk is enriched with document-level context via DeepSeek before indexing
- 🔍 **Hybrid Retrieval** — Dense vector similarity search via Qdrant
- 🎯 **Cross-Encoder Reranking** — BGE-Reranker-v2-M3 reranks top-20 candidates for precision
- 💭 **Chain-of-Thought Generation** — DeepSeek LLM generates reasoned answers with thinking traces
- 🖥️ **Web UI** — Drag-and-drop PDF upload, real-time Q&A, collapsible reasoning display
- 🔌 **REST API** — Programmatic access via `/upload-pdf` and `/ask` endpoints

## 📋 Prerequisites

- 🐍 **Python** 3.10+
- 🐳 **Docker** (for Qdrant)
- 🔑 **DeepSeek API Key** — Get one at [platform.deepseek.com](https://platform.deepseek.com/)

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rag.git
cd rag
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
QDRANT_URL=http://localhost:6333
```

### 5. Start Qdrant (Docker)

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 6. Run the application 🎉

**With Web UI (recommended):**

```bash
python ui.py
```

Opens at → [http://localhost:8000](http://localhost:8000)

**API-only mode (headless):**

```bash
python main.py
```

Swagger docs at → [http://localhost:8000/docs](http://localhost:8000/docs)

## 💡 Usage

### 🖥️ Web UI

1. 🌐 Open [http://localhost:8000](http://localhost:8000)
2. 📄 Upload a PDF using the drag-and-drop uploader
3. ❓ Type a question and click **Ask**
4. ✅ View the answer and expand the reasoning trace

### 🔌 REST API

**Upload a PDF:**

```bash
curl -X POST http://localhost:8000/upload-pdf \
  -F "file=@document.pdf"
```

**Ask a question:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'
```

**Response format:**

```json
{
  "query": "What are the key findings?",
  "reasoning": "Based on the context provided...",
  "answer": "The key findings include...",
  "sources_used": 5
}
```

## 📁 Project Structure

```
rag/
├── main.py              # FastAPI-only entry point (headless API)
├── ui.py                # NiceGUI + FastAPI entry point (Web UI + API)
├── config.py            # Settings & environment config (Pydantic)
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not committed)
└── src/
    ├── __init__.py
    ├── ingestion.py     # PDF parsing, contextual chunking, vector indexing
    ├── retrieval.py     # Hybrid search, cross-encoder reranking
    └── generation.py    # DeepSeek LLM answer generation
```

## 🔬 How It Works

### 📥 Ingestion Pipeline

1. **PDF Extraction** — Text is extracted page-by-page using `pypdf`
2. **Contextual Chunking** — Each page chunk is sent to DeepSeek along with the full document summary to generate a contextual header (inspired by Anthropic's contextual retrieval)
3. **Embedding** — The enriched chunk is encoded into a dense vector using `BAAI/bge-m3`
4. **Indexing** — Vectors are upserted into Qdrant with metadata (page number, context, original text)

### 🔎 Query Pipeline

1. **Embedding** — The user query is encoded using the same `BAAI/bge-m3` model
2. **Retrieval** — Top-20 candidate chunks are retrieved from Qdrant via cosine similarity
3. **Reranking** — A cross-encoder (`BAAI/bge-reranker-v2-m3`) rescores all candidates for precision
4. **Generation** — Top-5 reranked chunks are sent to DeepSeek LLM with the query for chain-of-thought answer generation

## ⚡ Configuration

All settings are managed via `config.py` and can be overridden with environment variables:

| Variable           | Default                   | Description                  |
| ------------------ | ------------------------- | ---------------------------- |
| `DEEPSEEK_API_KEY` | *(required)*              | DeepSeek API key             |
| `QDRANT_URL`       | `http://localhost:6333`   | Qdrant server URL            |
| `COLLECTION_NAME`  | `gold_standard_rag`       | Qdrant collection name       |
| `DENSE_MODEL`      | `BAAI/bge-m3`             | Sentence transformer model   |
| `RERANK_MODEL`     | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker model |

## 📄 License

MIT License

Ayman Sayed © [aymanAI.com](https://aymanai.com) Copyright 2026


