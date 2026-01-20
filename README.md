# 📈 Financial Analyst Agent — Tool-Augmented Agentic RAG with LangGraph (Cloud Run Ready)

![Status](https://img.shields.io/badge/Status-Deployed_on_Cloud_Run-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Tech](https://img.shields.io/badge/Stack-LangGraph_|_LangChain_|_Chroma_|_GCP-orange)

A production-oriented **Agentic RAG** system for grounded Q&A over SEC 10-K filings.  
It combines **Hybrid Retrieval (BM25 + Vector)** with **tool-enforced computation** (no mental math), and is containerized for **Google Cloud Run** deployment.

---

## 🎥 Project Demo

> Short demos showcasing retrieval, tool-based computation, and multi-turn reasoning.

### 1) Core Capability: RAG + Precision Math (Tool-Enforced)
**Scenario:** Query Apple’s 2024 Form 10-K (retrieval), then do a forward-looking projection based on retrieved net sales.

▶️ Demo:
https://github.com/denis7-jean/Financial-Analyst-Agent-RAG-LangGraph/releases/download/v1.0/demo_rag_calculation.mp4

### 2) Multi-turn Context: Follow-up Without Re-retrieval
**Scenario:** Compare the projected net sales against Apple’s 2023 historical data using prior context.

▶️ Demo:
https://github.com/denis7-jean/Financial-Analyst-Agent-RAG-LangGraph/releases/download/v1.0/demo_multiturn_comparison.mp4

---

## 📖 Project Overview

This system is designed to solve two common pain points in financial LLM apps:

1) **Hallucination** → answers must be grounded in filings (citations with source/page/chunk_id)  
2) **Math errors** → calculations are delegated to a deterministic tool (`calculator`)

Unlike a “simple RAG,” the agent is orchestrated by **LangGraph** and can decide when to:
- retrieve evidence (`search_10k`)
- compute deterministically (`calculator`)

---

## 🧠 High-Level Architecture

```mermaid
graph LR
    A[User Query] --> B[LangGraph Agent Node]
    B --> C{Tool Decision}
    C -- Retrieve --> D[search_10k: Hybrid Retriever]
    C -- Compute --> E[calculator: Deterministic Math]
    D --> F[State Update]
    E --> F[State Update]
    F --> G{Answer Ready?}
    G -- No --> B
    G -- Yes --> H[Final Answer + Citations]
````

---

## ✨ Key Features

### 1) Hybrid Retrieval (BM25 + Vector)

* **BM25** improves precision on financial terminology / exact phrases / tables
* **Vector search** improves semantic recall
* Results are merged by stable keys (`chunk_id` preferred) and fused with weighted normalized scores.

✅ Debug-friendly evidence formatting includes:

* `source / page / chunk_id`
* `bm25 / vec / final` scores (normalized fusion)

Example:

```
[S1] source: Apple_2024_10k.pdf | page: 37 | chunk_id: 184 | bm25=1.000 | vec=0.980 | final=0.988
```

### 2) Tool-Enforced Computation (No Mental Math)

* Any projection / YoY growth / differences are computed by the `calculator` tool
* The agent shows formula + numeric result explicitly

### 3) Cloud-Deployable (Docker + Cloud Run)

* Streamlit UI packaged into a Docker image
* Deployed to **Google Cloud Run**
* Secrets injected via env vars (e.g., `GOOGLE_API_KEY`)

---

## 🛠️ Tech Stack (Current)

* **Orchestration:** LangGraph, LangChain
* **LLM:** Google AI Studio (Gemini via `GOOGLE_API_KEY`, e.g. `gemini-flash-latest`)
* **Embeddings:** Vertex AI Embeddings (`text-embedding-004`)
* **Vector DB:** ChromaDB (persistent)
* **Retrieval:** Hybrid (BM25 + Vector)
* **Frontend:** Streamlit
* **Cloud:** Google Cloud Run, Cloud Storage (optional artifact sync)

> Note: Some sandbox GCP projects restrict Vertex GenAI models (404 access).
> This repo uses **AI Studio API key for LLM** while keeping embeddings/storage compatible with GCP.

---

## 📂 Project Structure

```bash
├── data/                   # Raw 10-K PDFs
├── vector_db/              # Local persisted Chroma artifacts (baked into Docker image)
├── src/
│   ├── ingestion/          # PDF loading, chunking, embedding, persistence
│   ├── retrieval/          # Chroma loading + BM25 + hybrid fusion
│   ├── graph/              # LangGraph nodes/edges/state
│   ├── tools/              # Tool wrappers (search_10k, calculator)
│   └── utils/              # Helpers
├── app.py                  # Streamlit entry
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started (Local)

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Set environment variables

Create `.env` (or export env vars in your shell):

```bash
# LLM (AI Studio)
GOOGLE_API_KEY=AIza...

# Optional: if you re-run ingestion with Vertex AI Embeddings
GCP_PROJECT_ID=your-project
GCP_REGION=us-central1
```

### 3) (Optional) Run ingestion

If you want to rebuild the vector store from PDFs:

```bash
python -m src.ingestion.ingest
```

### 4) Run the app

```bash
streamlit run app.py
```

---

## ☁️ Deploy to Google Cloud Run (Artifact Registry)

### 1) Enable required services

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com
```

### 2) Create a Docker repository (once)

```bash
gcloud artifacts repositories create fin-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker repository for fin-agent"
```

### 3) Build & push image

```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/fin-repo/fin-agent .
```

### 4) Deploy

```bash
gcloud run deploy fin-agent \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/fin-repo/fin-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=YOUR_AI_STUDIO_KEY
```

---

## 🧭 Roadmap (Next Tools)

* `web_search` (real-time news / sentiment)
* `get_stock_price` (yfinance / market data)
* richer analytics tools (time-series, ratio analysis)
* evaluation signals for retrieval quality & groundedness

---

**Author:** Huiyao Lan — MEng @ University of Toronto
This repo is part of an applied LLM engineering portfolio.

```
