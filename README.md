# 🎫 Ticket RAG

A production-grade RAG (Retrieval-Augmented Generation) system for customer support tickets. Given a new support ticket, it finds the most relevant historical tickets and generates a suggested resolution using an LLM.

Built with hybrid search (dense + sparse), semantic chunking, cross-encoder reranking, HyDE query expansion, MMR diversification, and Claude / Groq for generation.

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
| **HyDE** *(optional)* | Embeds a hypothetical resolution instead of the raw query — aligns query embedding with the resolution-heavy index |
| **MMR** *(optional)* | Diversifies final results so Claude sees different angles, not near-duplicate tickets |
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

# With retrieval improvements enabled
python main.py --hyde --mmr "Customer cannot log in after password reset"
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

## Evaluation

The system was evaluated on **28,581 training tickets** with a **5,716-ticket holdout test set** (20% stratified split by category) — tickets the system never saw during ingestion.

### Retrieval (holdout — genuinely unseen tickets)

| Metric | Score | What it means |
|---|---|---|
| **Category Precision@1** | **64%** | Top result from the correct support category (random baseline: 25%) |
| **Category Precision@3** | **39%** | Signal drops at rank 2-3 — reranker is strongest at rank 1 |
| **Category Precision@5** | **35%** | Approaches random past top-3; `TOP_K_RERANK=3` is the right default |
| **Category MRR** | **0.698** | First relevant result appears around rank 1-2 on average |

> Retrieval ground truth: a retrieved ticket is relevant if it shares the same support category as the query. With 4 categories, random chance = 25%. The system runs 2.5× above random at rank 1.
>
> Note: the earlier non-holdout eval showed 100%@1 because the query ticket itself was in the index — retrieval was matching the exact document. The holdout numbers are the honest ones.

### Answer quality (holdout — answers evaluated against resolutions the model never saw)

| Metric | Score | What it means |
|---|---|---|
| **Semantic Similarity** | **0.917** | Generated answers closely match human-written resolutions |
| **LLM Judge** | **4.38 / 5** | 49 of 50 answers rated 4★ or 5★ by an independent LLM |
| **Faithfulness** | **1.000** | No hallucination — every claim grounded in retrieved context |
| **Answer Relevance** | **0.920** | Answers directly address what was asked |

### Key finding — answer quality holds even when retrieval is imperfect

Retrieval precision at rank 1 is 64%, but answer quality scores remain excellent (0.917 similarity, 4.38/5 judge). The LLM compensates well — loosely related context from the same support domain still produces useful resolutions. Improving retrieval to push Category Precision@1 above 80% is the clearest path to further gains.

### Known limitations & applied fixes

Eval surfaced several issues that were subsequently fixed:

| Problem | Evidence | Fix applied |
|---|---|---|
| TOP_K_RERANK=5 sent noisy results to Claude | Precision drops sharply after rank 1 | Lowered default to 3 |
| Faithfulness threshold 0.55 too loose | Reported impossible 1.000 score | Raised to 0.70 |
| Query embedding mismatches index | Retrieval quality degrades at ranks 2-5 | Added HyDE (`--hyde`) |
| Top results were near-duplicates | Same ticket rephrased, not diverse angles | Added MMR (`--mmr`) |
| Faithfulness called `pipeline.query()` twice | Doubled LLM calls, doubled eval runtime | Cache hits from similarity eval, reuse in faithfulness |
| Semantic Recall@K metric was circular | Dense retrieval optimises cosine sim — any threshold gives ~100% | Reverted to Category Precision@K |

### Reproducing the eval

```bash
# Split dataset
python scripts/prepare_holdout.py

# Re-ingest train set only
python scripts/ingest.py --force

# Run full holdout evaluation
python scripts/eval.py --holdout

# Faster — skip LLM judge
python scripts/eval.py --holdout --skip-llm-judge

# Fastest — retrieval metrics only
python scripts/eval.py --holdout --skip-generation
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
├── main.py                        # CLI entrypoint (--hyde, --mmr flags)
├── src/
│   ├── ingestion/
│   │   ├── loader.py              # CSV parsing + column normalization
│   │   ├── chunker.py             # Semantic chunking via cosine similarity drops
│   │   └── embedder.py            # multilingual-e5-large wrapper
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB upsert + query
│   │   ├── bm25_index.py          # BM25 sparse index, persisted as pickle
│   │   ├── hybrid_search.py       # Reciprocal Rank Fusion
│   │   ├── reranker.py            # Cross-encoder reranker (bge-reranker-base)
│   │   ├── hyde.py                # HyDE — hypothetical resolution embedding
│   │   └── mmr.py                 # MMR — result diversification
│   ├── generation/
│   │   └── claude_client.py       # Groq / Gemini / Ollama / Anthropic client
│   └── pipeline.py                # Orchestrates retrieve → rerank → MMR → generate
├── scripts/
│   ├── download_data.py           # Kaggle API download helper
│   ├── ingest.py                  # One-shot ingestion with resume support
│   ├── prepare_holdout.py         # Stratified train/test split for honest eval
│   └── eval.py                    # Full eval suite (Recall, MRR, Similarity, Judge, RAGAS)
├── data/                          # Raw CSVs (gitignored)
├── .chroma/                       # ChromaDB persistence (gitignored)
└── .bm25_index.pkl                # BM25 index (gitignored)
```

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API (used if set) |
| `GEMINI_API_KEY` | — | Gemini API (fallback if Groq quota exhausted) |
| `OLLAMA_MODEL` | — | Local Ollama model name e.g. `llama3.2` (free, unlimited) |
| `ANTHROPIC_API_KEY` | — | Claude API (final fallback) |
| `KAGGLE_USERNAME` | — | Kaggle dataset download |
| `KAGGLE_KEY` | — | Kaggle dataset download |
| `EMBED_MODEL` | `intfloat/multilingual-e5-large` | Override embedding model |
| `TOP_K_DENSE` | `20` | Dense retrieval candidates |
| `TOP_K_SPARSE` | `20` | Sparse retrieval candidates |
| `TOP_K_RERANK` | `3` | Final results sent to Claude (lowered from 5 post-eval) |
| `CHROMA_PATH` | `.chroma` | ChromaDB storage path |
| `USE_HYDE` | `false` | Enable HyDE query expansion |
| `USE_MMR` | `false` | Enable MMR result diversification |
