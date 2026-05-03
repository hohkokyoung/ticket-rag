# ticket-rag

A production-grade RAG system for customer support tickets. Given a new ticket, it retrieves the most relevant historical tickets and generates suggested resolutions using Claude.

## Architecture

```
New Ticket
    │
    ▼
Embedding (multilingual-e5-large)
    │
    ├──► Dense Search (ChromaDB)  ──┐
    │                               ├──► RRF Fusion ──► Cross-Encoder Rerank ──► Claude Generation
    └──► Sparse Search (BM25)    ──┘
```

**Key design choices:**
- **Multilingual embeddings** (`intfloat/multilingual-e5-large`) — matches the dataset's 10+ languages
- **Hybrid search** — dense (semantic) + sparse (keyword BM25) via Reciprocal Rank Fusion
- **Semantic chunking** — splits long tickets at semantic boundary drops, not fixed token windows
- **Cross-encoder reranking** (`BAAI/bge-reranker-base`) — re-scores top-k candidates with full query-passage context
- **Claude claude-sonnet-4-6** — generates solutions with cited source tickets

## Dataset

Kaggle: `tobiasbueck/multilingual-customer-support-tickets`

Columns used: `ticket_id`, `subject`, `body`, `resolution`, `category`, `language`, `priority`

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and KAGGLE_USERNAME + KAGGLE_KEY
```

### 3. Download data
```bash
python scripts/download_data.py
```

### 4. Ingest (embed + index) — takes ~10-20 min first run, cached after
```bash
python scripts/ingest.py
```

### 5. Query
```bash
python main.py "Customer cannot log in after password reset"
# or interactive mode:
python main.py
```

## Project structure

```
ticket-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py       # CSV parsing + field normalization
│   │   ├── chunker.py      # Semantic chunking via embedding cosine similarity
│   │   └── embedder.py     # multilingual-e5-large wrapper
│   ├── retrieval/
│   │   ├── vector_store.py # ChromaDB CRUD
│   │   ├── bm25_index.py   # BM25 via rank_bm25, persisted as pickle
│   │   ├── hybrid_search.py# RRF fusion of dense + sparse results
│   │   └── reranker.py     # Cross-encoder reranker (bge-reranker-base)
│   ├── generation/
│   │   └── claude_client.py# Anthropic SDK call with prompt caching
│   └── pipeline.py         # Orchestrates retrieve → rerank → generate
├── scripts/
│   ├── download_data.py    # Kaggle API download helper
│   └── ingest.py           # One-shot ingestion runner
├── data/                   # Raw CSVs (gitignored)
├── .chroma/                # ChromaDB persistence (gitignored)
├── .bm25_index.pkl         # BM25 pickle (gitignored)
├── requirements.txt
└── main.py                 # CLI entrypoint
```

## Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API access |
| `KAGGLE_USERNAME` | Kaggle credentials |
| `KAGGLE_KEY` | Kaggle API key |
| `EMBED_MODEL` | Override embedding model (default: `intfloat/multilingual-e5-large`) |
| `TOP_K_DENSE` | Dense retrieval candidates (default: 20) |
| `TOP_K_SPARSE` | Sparse retrieval candidates (default: 20) |
| `TOP_K_RERANK` | Final results after reranking (default: 5) |
| `CHROMA_PATH` | ChromaDB storage path (default: `.chroma`) |
