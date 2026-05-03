# 🎫 Ticket RAG

A production-grade RAG (Retrieval-Augmented Generation) system for customer support tickets. Given a new support ticket, it finds the most relevant historical tickets and generates a suggested resolution using an LLM.

Built with hybrid search (dense + sparse), semantic chunking, cross-encoder reranking, and Claude / Groq for generation.

---

## How it works

```
New Ticket
    │
    ▼
Embed (multilingual-e5-large)
    │
    ├──► Dense Search (ChromaDB)  ──┐
    │                               ├──► RRF Fusion ──► Cross-Encoder Rerank ──► LLM Generation
    └──► Sparse Search (BM25)    ──┘
```

**Why each layer exists:**

| Layer | Why |
|---|---|
| **Multilingual embeddings** | Dataset spans 10+ languages — a multilingual model matches across them |
| **Hybrid search (Dense + BM25)** | Dense catches paraphrases; BM25 catches exact keywords like error codes |
| **RRF fusion** | Combines both ranked lists without needing to tune score weights |
| **Cross-encoder reranking** | Re-scores candidates by seeing query + passage together — far more accurate than cosine alone |
| **Semantic chunking** | Splits long tickets at meaning boundaries, not fixed token windows |
| **Prompt caching** | Reduces LLM cost by caching the system prompt and retrieved context |

---

## Tech stack

- **Embeddings** — `intfloat/multilingual-e5-large` via `sentence-transformers`
- **Vector store** — ChromaDB (local, persistent)
- **Sparse search** — BM25 via `rank_bm25`
- **Reranker** — `BAAI/bge-reranker-base` cross-encoder
- **Generation** — Groq (Llama 3.3 70B, free tier) or Anthropic Claude
- **Dataset** — [Kaggle: Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get your API keys

**Groq** (free) — https://console.groq.com → API Keys

**Kaggle** — https://www.kaggle.com/settings → API → Create New Token

### 3. Configure environment
```bash
cp .env.example .env
```

Fill in `.env`:
```env
GROQ_API_KEY=gsk_...
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_key
```

### 4. Download the dataset
```bash
python scripts/download_data.py
```

### 5. Ingest — embed and index all tickets
```bash
python scripts/ingest.py
```

> First run downloads `multilingual-e5-large` (~2.3 GB) and embeds all tickets. Takes 10–20 min. Fully cached after — subsequent startups are instant.

### 6. Query
```bash
# Single query
python main.py "Customer cannot log in after password reset"

# Interactive mode
python main.py

# Retrieval only — skip generation, just see matched tickets
python main.py --retrieve-only "billing charge incorrect"
```

---

## Sample output

```
Query: Customer cannot log in after password reset

  #   Ticket ID      Category          Lang   Rerank Score   Preview
 ─────────────────────────────────────────────────────────────────────────────────
  1   ticket_00412   Account Access    en     0.9821         Subject: Cannot log in after password reset  Problem: User re…
  2   ticket_00089   Authentication    en     0.9134         Subject: Password reset link not working  Problem: Customer cl…
  3   ticket_01203   Account Access    de     0.8876         Subject: Anmeldung nach Passwort-Reset fehlgeschlagen  Problem…
  4   ticket_00731   Security          en     0.7654         Subject: Account locked after multiple login attempts  Problem…
  5   ticket_00956   Account Access    fr     0.7201         Subject: Impossible de se connecter  Problem: L'utilisateur si…

╭─────────────────────────── Suggested Resolution ────────────────────────────────╮
│                                                                                  │
│  Based on similar cases (tickets #00412, #00089, #01203), here is the           │
│  recommended resolution:                                                         │
│                                                                                  │
│  **Immediate steps:**                                                            │
│  1. Ask the customer to clear browser cache and cookies, then retry              │
│  2. Confirm the reset link was used within 15 minutes (links expire)             │
│  3. Check whether the customer is logging in with the correct email address      │
│     (some customers have multiple accounts)                                      │
│                                                                                  │
│  **If the above doesn't resolve it:**                                            │
│  4. Manually invalidate all active sessions from the admin panel                 │
│  5. Force-expire the current password and issue a new reset link                 │
│  6. If the account shows as locked, unlock it before sending the reset           │
│                                                                                  │
│  **Note:** Ticket #01203 (German) had the same issue caused by a cached          │
│  browser session holding an old auth token. Step 4 resolved it immediately.      │
│                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

---

## Resume support

Ingestion checkpoints after every batch. If interrupted, re-running continues from where it stopped:

```bash
python scripts/ingest.py
# Resuming from checkpoint: 8432 chunks, 12 batches already embedded.
# Embedding 21/33 batches (12 already done)...

# To start fresh:
python scripts/ingest.py --force
```

---

## Project structure

```
ticket-rag/
├── main.py                        # CLI entrypoint
├── src/
│   ├── ingestion/
│   │   ├── loader.py              # CSV parsing + column normalization
│   │   ├── chunker.py             # Semantic chunking via cosine similarity drops
│   │   └── embedder.py            # multilingual-e5-large wrapper
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB upsert + query
│   │   ├── bm25_index.py          # BM25 sparse index, persisted as JSON
│   │   ├── hybrid_search.py       # Reciprocal Rank Fusion
│   │   └── reranker.py            # Cross-encoder reranker
│   ├── generation/
│   │   └── claude_client.py       # Groq / Anthropic generation client
│   └── pipeline.py                # Orchestrates retrieve → rerank → generate
├── scripts/
│   ├── download_data.py           # Kaggle API download helper
│   └── ingest.py                  # One-shot ingestion runner with resume support
├── data/                          # Raw CSVs (gitignored)
├── .chroma/                       # ChromaDB persistence (gitignored)
└── .bm25_index.pkl                # BM25 index (gitignored)
```

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API (used if set) |
| `ANTHROPIC_API_KEY` | — | Claude API (fallback if no Groq key) |
| `KAGGLE_USERNAME` | — | Kaggle dataset download |
| `KAGGLE_KEY` | — | Kaggle dataset download |
| `EMBED_MODEL` | `intfloat/multilingual-e5-large` | Override embedding model |
| `TOP_K_DENSE` | `20` | Dense retrieval candidates |
| `TOP_K_SPARSE` | `20` | Sparse retrieval candidates |
| `TOP_K_RERANK` | `5` | Final results after reranking |
| `CHROMA_PATH` | `.chroma` | ChromaDB storage path |
